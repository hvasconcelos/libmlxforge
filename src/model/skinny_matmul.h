// Multi-row GEMV kernels for the batched-decode regime.
//
// MLX's Metal matmul drops from ~161 GB/s (GEMV, M=1) to ~56 GB/s (tiled GEMM)
// the moment M reaches 2, and every M in [2, 64] pays the same tile cost
// (ml-explore/mlx#3661) — exactly the continuous-batching decode shape, where
// each per-token step is weight-bandwidth-bound. These custom
// fast::metal_kernel kernels read each weight row once per simdgroup and keep
// the M activation rows as register accumulators, recovering GEMV-class
// bandwidth: a one-column-per-simdgroup variant for M in [2, 4] (~161 GB/s)
// and a two-column variant for M in [5, 16] (the doubled arithmetic intensity
// halves the redundant activation reads that degrade larger M; ~125 GB/s at
// M=8, still ahead of the tiled GEMM at M=16). Beyond 16 the fallback GEMM is
// the right path.
//
// Accumulation is fp32 but in a different order than mx::matmul, so logits can
// differ at fp16-noise scale — the same class as the decode-vs-recompute gap.
// Gated by EngineConfig::skinny_mm (default on) via DecoderModel::set_skinny_mm
// and token-equality tests against the stock-matmul stream.
#pragma once

#include "mlx/array.h"

namespace mx = mlx::core;

namespace mlxforge {

// True when the kernel path applies to the shapes: x is (B, 1, D) or (B, D)
// fp16 with B in [2, 16], w is a dense fp16 (O, D) weight, and D is a multiple
// of 128 (half4 loads across 32 lanes). Enablement is the caller's flag
// (DecoderModel::skinny_mm_); this checks shapes only.
bool skinny_matmul_applies(const mx::array& x, const mx::array& w);

// x @ w.T via the multi-row GEMV kernels. Preserves x's leading shape:
// (B, 1, D) -> (B, 1, O), (B, D) -> (B, O).
mx::array skinny_matmul(const mx::array& x, const mx::array& w);

}  // namespace mlxforge
