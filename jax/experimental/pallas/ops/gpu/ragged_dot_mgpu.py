# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Ragged dot Pallas-Mosaic-GPU implementation."""

import dataclasses
import functools
import itertools
import math
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax._src import test_util as jtu  # noqa: F401
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import profiler
from jax.experimental.pallas import mosaic_gpu as plgpu
import numpy as np


@dataclasses.dataclass(frozen=True)
class GroupInfo:
  """Information regarding the group being processed in a block."""

  group_id: jax.Array
  block: jax.Array
  block_start: jax.Array
  actual_start: jax.Array
  actual_end: jax.Array
  start_within_block: jax.Array
  actual_size: jax.Array

  @classmethod
  def create(cls, group_lengths, tile, tid):
    """Get the group info for the current block."""

    tile = jnp.int32(tile)
    group_boundaries = [group_lengths[i] for i in range(group_lengths.shape[0])]

    # We usually only have very few groups, so we unroll the loop processing
    # them. Normally we'd break out of the loop early, once we'd have found our
    # boundary, but we can't do that when unrolling, so we rely on many selects
    # to mask out the epilogue of the loop.
    group_end = group_start = block = group = end = jnp.array(
        0, dtype=jnp.int32
    )

    for i, b in enumerate(group_boundaries):
      # Start/end are inclusive
      start = end
      end = start + b
      final = end - 1
      start_block = lax.div(start, tile)
      final_block = lax.div(final, tile)
      block_end = final_block + 1
      tid_begin = start_block + i
      tid_end = block_end + i
      # How many blocks after is our block?
      this_is_group = (tid_begin <= tid) & (tid < tid_end)
      block = lax.select(this_is_group, tid - tid_begin + start_block, block)
      group = lax.select(this_is_group, jnp.int32(i), group)
      group_start = lax.select(this_is_group, start, group_start)
      group_end = lax.select(this_is_group, end, group_end)

    block_start = block * tile
    actual_start = jnp.maximum(group_start, block_start)
    actual_end = jnp.minimum(group_end, block_start + tile)
    start_within_block = actual_start - block_start
    actual_size = actual_end - actual_start
    return cls(
        group_id=group,
        block=block,
        block_start=block_start,
        actual_start=actual_start,
        actual_end=actual_end,
        start_within_block=start_within_block,
        actual_size=actual_size,
    )


def _find_swizzle(dim_size_bits: int, what: str):
  for swizzle_bytes in (128, 64, 32, 16):
    if dim_size_bits % (swizzle_bytes * 8) == 0:
      return swizzle_bytes
  raise ValueError(
      f"No valid out swizzle for {what}: its minor dimension has"
      f" {dim_size_bits} bits, which is not a multiple of 128"
  )


def ragged_dot(
    lhs,  # (M, K)
    rhs,  # (G, K, N)
    *,
    group_sizes,  # (G,)
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array:
  if lhs.dtype != rhs.dtype:
    raise NotImplementedError(
        f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}"
    )

  elem_bits = jnp.finfo(lhs.dtype).bits
  swizzle = _find_swizzle(elem_bits * block_k, "lhs")
  swizzle_elems = swizzle * 8 // elem_bits

  (m, k) = lhs.shape
  (g, k2, n) = rhs.shape

  if k != k2:
    raise ValueError(f"lhs.shape={k} must match rhs.shape={k2}")

  if k % block_k != 0:
    raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

  num_sms = 132

  def body(rows_per_expert_gmem, lhs_gmem, rhs_gmem, o_gmem):
    grid = (
        grid_block_n,
        pl.cdiv(m, block_m) + g - 1,
        pl.cdiv(n, grid_block_n * block_n),
    )
    grid_rev = grid[::-1]
    grid_size = math.prod(grid)

    def mn_loop(step, _):
      step = step * num_sms + sm
      idx = []
      for i in range(len(grid_rev)):
        idx.append(
            lax.rem(
                lax.div(step, jnp.int32(math.prod(grid_rev[i + 1 :]))),
                jnp.int32(grid_rev[i]),
            )
        )

      block_ni, mi, remainder_ni = idx[::-1]  # pylint: disable=unbalanced-tuple-unpacking
      ni = block_ni * pl.cdiv(n, block_n * grid_block_n) + remainder_ni
      group_info = GroupInfo.create(rows_per_expert_gmem, block_m, mi)

      def acc_scope(acc_ref):
        def compute(_, lhs_smem, rhs_smem):
          plgpu.wgmma(acc_ref, lhs_smem, rhs_smem)
        plgpu.emit_pipeline(
            compute,
            grid=(k // block_k,),
            in_specs=[
                plgpu.GPUBlockSpec(
                    (block_m, block_k),
                    lambda k: (group_info.block, k),
                    transforms=transforms,
                ),
                plgpu.GPUBlockSpec(
                    (block_k, block_n), lambda k: (k, ni), transforms=transforms
                ),
            ],
            max_concurrent_steps=max_concurrent_steps,
            delay_release=1,
        )(lhs_gmem, rhs_gmem.at[group_info.group_id])
        return acc_ref[...]

      acc = pl.run_scoped(acc_scope, plgpu.ACC((block_m, block_n)))

      def store_scope(o_smem):
        o_smem[...] = acc.astype(o_smem.dtype)
        plgpu.commit_smem()

        smem_start = group_info.start_within_block

        # pylint: disable=cell-var-from-loop
        remaining_rows = min(block_m, m)
        while remaining_rows > 0:
          const_rows_len = 1 << int(math.log2(remaining_rows))
          remaining_rows //= 2

          @pl.when(group_info.actual_size & const_rows_len != 0)
          def _():
            o_smem_slice = o_smem.at[pl.ds(smem_start, const_rows_len)]
            o_gref_slice = o_gmem.at[
                pl.ds(group_info.block_start + smem_start, const_rows_len),
                pl.ds(ni * block_n, block_n),
            ]
            plgpu.copy_smem_to_gmem(o_smem_slice, o_gref_slice)

          smem_start += group_info.actual_size & const_rows_len
        plgpu.wait_smem_to_gmem(0, wait_read_only=True)

      pl.run_scoped(
          store_scope,
          plgpu.SMEM(
              (block_m, block_n),
              dtype=o_gmem.dtype,
              transforms=(
                  plgpu.TilingTransform((1, swizzle_elems)),
                  plgpu.SwizzleTransform(swizzle),
              ),
          ),
      )

    sm = lax.axis_index("block")
    num_steps = (grid_size // num_sms) + (sm < grid_size % num_sms).astype(
        jnp.int32
    )
    jax.lax.fori_loop(0, num_steps, mn_loop, None)

  transforms = (
      plgpu.TilingTransform((8, swizzle_elems)),
      plgpu.SwizzleTransform(swizzle),
  )

  kernel = plgpu.kernel(
      body,
      out_shape=jax.ShapeDtypeStruct((m, n), lhs.dtype),
      grid=(num_sms,),
      grid_names=("block",),
  )
  return kernel(group_sizes, lhs, rhs)


def main(unused_argv):
  m, k, n, num_groups = 16 * 1024, 2048, 16 * 1024, 16
  kx, ky, kz = random.split(random.key(1234), num=3)

  lhs = jax.random.normal(kx, (m, k), jnp.float16)
  rhs = jax.random.normal(ky, (num_groups, k, n), jnp.float16)
  group_boundaries = jax.lax.sort(
      jax.random.randint(kz, (num_groups - 1,), 0, m, jnp.int32)
  )
  group_starts = lax.concatenate(
      [jnp.array([0], dtype=jnp.int32), group_boundaries], 0
  )
  group_ends = lax.concatenate(
      [group_boundaries, jnp.array([m], dtype=jnp.int32)], 0
  )
  group_sizes = group_ends - group_starts
  assert group_sizes.shape == (num_groups,)

  block_m = block_n = (64, 128, 192)
  block_k = (64,)
  max_concurrent_steps = (2, 4, 5, 6)
  grid_block_n = (1, 2, 4, 8, 16)
  configs = itertools.product(
      block_m, block_n, block_k, max_concurrent_steps, grid_block_n
  )
  names = (
      "block_m", "block_n", "block_k", "max_concurrent_steps", "grid_block_n"
  )
  best_runtime = float("inf")
  best_kwargs = {}
  for config in configs:
    kwargs = dict(zip(names, config))
    if n % (kwargs["grid_block_n"] * kwargs["block_n"]):
      continue
    try:
      f = functools.partial(ragged_dot, group_sizes=group_sizes, **kwargs)
      _, runtime = profiler.measure(f, mode="cupti")(lhs, rhs)
    except ValueError as e:
      if "Mosaic GPU kernel exceeds available shared memory" not in str(e):
        raise
      runtime = float("inf")
    # Enable this to get more detailed information.
    else:
      print(" ".join(f"{k}={v}" for k, v in kwargs.items()), int(runtime * 1000))
    if runtime < best_runtime:  # pytype: disable=unsupported-operands
      best_runtime = runtime
      best_kwargs = kwargs
  if not best_kwargs:
    raise ValueError("No valid configuration found")

  ref, ref_runtime = profiler.measure(jax.lax.ragged_dot)(
      lhs, rhs, group_sizes=group_sizes
  )
  result = ragged_dot(lhs, rhs, group_sizes=group_sizes, **best_kwargs)
  np.testing.assert_allclose(result, ref, atol=1e-3, rtol=1e-3)

  tflops = float(2 * k * m * n) / (best_runtime / 1e3) / 1e12
  ref_tflops = float(2 * k * m * n) / (ref_runtime / 1e3) / 1e12
  print(
      "Best parameters: ", " ".join(f"{k}={v}" for k, v in best_kwargs.items())
  )
  print(f"Kernel:    {best_runtime * 1000:.1f} us = {tflops:.1f} TFLOPS")
  print(f"Reference: {ref_runtime * 1000:.1f} us = {ref_tflops:.1f} TFLOPS")


if __name__ == "__main__":
  from absl import app

  jax.config.config_with_absl()
  app.run(main)
