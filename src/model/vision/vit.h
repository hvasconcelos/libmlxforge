// Qwen3-VL vision transformer (ViT) encoder.
//
// Consumes preprocessed image patches (pixel_values) + the patch grid (grid_thw)
// and produces the merged patch embeddings the language model attends over, plus
// the per-layer DeepStack features injected into the first decoder layers.
//
// Unlike the text decoder, the ViT weights are *unquantized* fp16 and every
// Linear / LayerNorm carries a bias. The encoder borrows the owning model's
// Weights (keys canonicalized to "visual.*" by sanitize_key) and is built up
// stage by stage, each gated against reference/fixtures_qwen3_vl:
//   patch_embed  -> vit_patch_embed   (Conv3d-as-matmul patchify)
//   pos_embed    -> vit_pos_embed     (interpolated learned position embeds)
//   rot_pos_emb  -> vit_rotary        (2D RoPE frequencies)
//   block(0)     -> vit_block0        (attention + MLP)
//   forward      -> vit_out, deepstack_{0,1,2}
#pragma once

#include "mlx/array.h"

#include "core/config.h"
#include "core/weights.h"

namespace mlxforge {

namespace mx = mlx::core;

class VitEncoder {
 public:
  // Borrows `weights` (the owning model outlives the encoder); copies the small
  // VisionConfig by value.
  VitEncoder(VisionConfig cfg, const Weights& weights);

  const VisionConfig& config() const { return cfg_; }

  // Conv3d patch embedding as a matmul: pixel_values (num_patches, in_ch *
  // temporal_patch * patch * patch) -> (num_patches, vit_hidden). The conv weight
  // is a full-receptive-field kernel, so it reduces to a linear projection over
  // the flattened patch (the processor lays the patch out in the same
  // channel/temporal/height/width order as the raw weight, so no reorder needed).
  mx::array patch_embed(const mx::array& pixel_values) const;

 private:
  // y = x @ W^T + b for a ViT Linear stored under "<key>.weight" / "<key>.bias".
  mx::array linear(const mx::array& x, const std::string& key) const;

  VisionConfig cfg_;
  const Weights& w_;
};

}  // namespace mlxforge
