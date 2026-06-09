#include "vision/preprocess.h"

#include <stdexcept>

#include "mlx/ops.h"

namespace mlxforge {

Preprocessed patchify_image(const mx::array& image_rgb, const PreprocessConfig& cfg) {
  const int H = image_rgb.shape()[0];
  const int W = image_rgb.shape()[1];
  const int C = image_rgb.shape()[2];
  const int ps = cfg.patch_size, tps = cfg.temporal_patch_size, ms = cfg.merge_size;
  const int factor = ps * ms;
  if (H % factor != 0 || W % factor != 0) {
    throw std::runtime_error("patchify_image: image dimensions must be a multiple of "
                             "patch_size*merge_size (resize is a separate stage)");
  }
  const int gh = H / ps, gw = W / ps;

  // Rescale (uint8 -> [0,1]) and per-channel normalize: (x*rescale - mean) / std.
  mx::array img = mx::multiply(mx::astype(image_rgb, mx::float32), mx::array(cfg.rescale_factor));
  mx::array mean(cfg.image_mean.data(), {1, 1, C}, mx::float32);
  mx::array std(cfg.image_std.data(), {1, 1, C}, mx::float32);
  img = mx::divide(mx::subtract(img, mean), std);  // (H, W, C)

  // HWC -> CHW, then duplicate along the temporal axis (a still image fills all
  // temporal_patch_size frames). Shape: (1, 1, tps, C, H, W).
  img = mx::transpose(img, {2, 0, 1});  // (C, H, W)
  img = mx::broadcast_to(mx::reshape(img, {1, 1, 1, C, H, W}), {1, 1, tps, C, H, W});

  // Split H -> (gh/ms, ms, ps) and W -> (gw/ms, ms, ps), then permute into
  // merged-block order with each row flattened as (C, tps, ps, ps). Mirrors the
  // HF Qwen3VLImageProcessor reshape/transpose exactly.
  img = mx::reshape(img, {1, 1, tps, C, gh / ms, ms, ps, gw / ms, ms, ps});
  img = mx::transpose(img, {0, 1, 4, 7, 5, 8, 3, 2, 6, 9});
  mx::array pixel_values = mx::reshape(img, {gh * gw, C * tps * ps * ps});

  return {pixel_values, {1, gh, gw}};
}

}  // namespace mlxforge
