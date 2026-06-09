// Engine-side image preprocessing for the Qwen3-VL ViT.
//
// Turns a decoded RGB image into the model-ready `pixel_values` + patch grid the
// ViT consumes. This first stage covers rescale + normalize + patchify for an
// image whose dimensions are already a multiple of patch_size*merge_size (the
// smart-resize stage — which must match HF's bicubic resampling — lands next).
//
// The patch layout mirrors the HF Qwen3VLImageProcessor exactly: rows are in
// merged-block order (the order the ViT's RoPE / position embeds assume) and each
// row is the patch flattened as (channel, temporal, patch_h, patch_w).
#pragma once

#include <array>

#include "mlx/array.h"

#include "core/config.h"

namespace mlxforge {

namespace mx = mlx::core;

// Image normalization / patchification parameters. patch/temporal/merge come
// from the model's VisionConfig; mean/std/rescale come from the HF
// preprocessor_config (Qwen3-VL defaults shown).
struct PreprocessConfig {
  int patch_size = 16;
  int temporal_patch_size = 2;
  int merge_size = 2;
  float rescale_factor = 1.0f / 255.0f;
  std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
  std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};

  // Patch/temporal/merge from the model config; normalization at Qwen3-VL defaults.
  static PreprocessConfig from(const VisionConfig& v) {
    PreprocessConfig c;
    c.patch_size = v.patch_size;
    c.temporal_patch_size = v.temporal_patch_size;
    c.merge_size = v.spatial_merge_size;
    return c;
  }
};

struct Preprocessed {
  mx::array pixel_values;        // (grid_h*grid_w, channels*tps*ps*ps) float32
  std::array<int, 3> grid_thw;   // (temporal, height, width) in patches
};

// Rescale + normalize + patchify an RGB image whose height and width are already
// multiples of patch_size*merge_size. `image_rgb` is (H, W, 3) uint8. Throws if
// the dimensions are not aligned (resize is a separate, later stage).
Preprocessed patchify_image(const mx::array& image_rgb, const PreprocessConfig& cfg);

}  // namespace mlxforge
