#include "model/skinny_matmul.h"

#include <optional>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mlxforge {

namespace {

// One simdgroup per output column o: 32 lanes split D with half4 loads, so the
// weight row is read once (coalesced) for all M activation rows; per-lane fp32
// accumulators reduce via simd_sum. M is a compile-time template arg so the
// accumulators stay in registers. Best for M in [2, 4].
constexpr const char* kSourceOneCol = R"(
    const uint lane = thread_position_in_grid.x;   // 0..31
    const uint o    = thread_position_in_grid.y;   // output column (w row)
    const int  D    = w_shape[1];
    const int  O    = w_shape[0];

    const device half4* wrow = (const device half4*)(w + (size_t)o * D);
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int k = lane; k < D / 4; k += 32) {
        half4 wv = wrow[k];
        for (int m = 0; m < M; ++m) {
            half4 xv = ((const device half4*)(x + (size_t)m * D))[k];
            acc[m] += (float)wv.x * (float)xv.x + (float)wv.y * (float)xv.y +
                      (float)wv.z * (float)xv.z + (float)wv.w * (float)xv.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float r = simd_sum(acc[m]);
        if (lane == 0) y[(size_t)m * O + o] = (half)r;
    }
)";

// Two output columns per simdgroup: each activation load feeds two weight
// rows, halving the redundant x traffic that degrades the one-column variant
// past M ~ 8. Best for M in [5, 16]; the crossover vs the tiled GEMM is past
// 16 (66 GB/s at M=16 vs the GEMM's flat ~56 on M1 Pro).
constexpr const char* kSourceTwoCol = R"(
    const uint lane = thread_position_in_grid.x;   // 0..31
    const uint pair = thread_position_in_grid.y;   // output column pair
    const int  D    = w_shape[1];
    const int  O    = w_shape[0];
    const uint o0   = pair * 2;
    const uint o1   = o0 + 1;
    const bool has1 = o1 < (uint)O;

    const device half4* w0 = (const device half4*)(w + (size_t)o0 * D);
    const device half4* w1 = (const device half4*)(w + (size_t)(has1 ? o1 : o0) * D);
    float acc0[M];
    float acc1[M];
    for (int m = 0; m < M; ++m) { acc0[m] = 0.0f; acc1[m] = 0.0f; }

    for (int k = lane; k < D / 4; k += 32) {
        half4 wv0 = w0[k];
        half4 wv1 = w1[k];
        for (int m = 0; m < M; ++m) {
            half4 xv = ((const device half4*)(x + (size_t)m * D))[k];
            acc0[m] += (float)wv0.x * (float)xv.x + (float)wv0.y * (float)xv.y +
                       (float)wv0.z * (float)xv.z + (float)wv0.w * (float)xv.w;
            acc1[m] += (float)wv1.x * (float)xv.x + (float)wv1.y * (float)xv.y +
                       (float)wv1.z * (float)xv.z + (float)wv1.w * (float)xv.w;
        }
    }
    for (int m = 0; m < M; ++m) {
        float r0 = simd_sum(acc0[m]);
        float r1 = simd_sum(acc1[m]);
        if (lane == 0) {
            y[(size_t)m * O + o0] = (half)r0;
            if (has1) y[(size_t)m * O + o1] = (half)r1;
        }
    }
)";

constexpr int kOneColMaxM = 4;
constexpr int kMaxM = 16;

}  // namespace

bool skinny_matmul_applies(const mx::array& x, const mx::array& w) {
  if (x.dtype() != mx::float16 || w.dtype() != mx::float16) return false;
  if (w.ndim() != 2 || w.shape()[1] % 128 != 0) return false;
  const int nd = x.ndim();
  if (nd == 3 && x.shape()[1] != 1) return false;  // decode shape only, never prefill
  if (nd != 2 && nd != 3) return false;
  const int m = x.shape()[0];
  return m >= 2 && m <= kMaxM && x.shape()[nd - 1] == w.shape()[1];
}

mx::array skinny_matmul(const mx::array& x, const mx::array& w) {
  static const auto one_col = mx::fast::metal_kernel(
      "mlxforge_gemv_multirow", {"x", "w"}, {"y"}, kSourceOneCol);
  static const auto two_col = mx::fast::metal_kernel(
      "mlxforge_gemv_multirow2", {"x", "w"}, {"y"}, kSourceTwoCol);

  const int m = x.shape()[0];
  const int o = w.shape()[0];
  mx::array x2 = x.ndim() == 3 ? mx::reshape(x, {m, x.shape()[2]}) : x;
  const bool narrow = m <= kOneColMaxM;
  std::vector<mx::array> out = (narrow ? one_col : two_col)(
      {x2, w}, {mx::Shape{m, o}}, {mx::float16},
      /*grid=*/{32, narrow ? o : (o + 1) / 2, 1}, /*threadgroup=*/{32, 1, 1},
      /*template_args=*/{{"M", m}},
      /*init_value=*/std::nullopt, /*verbose=*/false, {});
  return x.ndim() == 3 ? mx::reshape(out[0], {m, 1, o}) : out[0];
}

}  // namespace mlxforge
