#include "model/vision/vit.h"

#include "mlx/ops.h"

namespace mlxforge {

VitEncoder::VitEncoder(VisionConfig cfg, const Weights& weights)
    : cfg_(std::move(cfg)), w_(weights) {}

mx::array VitEncoder::linear(const mx::array& x, const std::string& key) const {
  // ViT weights are unquantized fp16; weight is (out, in), bias is (out,).
  mx::array y = mx::matmul(x, mx::transpose(w_.at(key + ".weight")));
  return mx::add(y, w_.at(key + ".bias"));
}

mx::array VitEncoder::patch_embed(const mx::array& pixel_values) const {
  // The Conv3d kernel spans a whole patch (kernel == stride == patch extent), so
  // it reduces to a linear projection of the flattened patch. The MLX-format
  // weight is (vit_hidden, temporal, patch, patch, in_ch), whereas the processor
  // lays each pixel_values row out as (in_ch, temporal, patch, patch); reorder the
  // patch to (temporal, patch, patch, in_ch) — the reference's moveaxis — so the
  // flattened axes line up, then matmul reproduces the convolution.
  const int P = pixel_values.shape()[0];  // num patches
  const int c = cfg_.in_channels, t = cfg_.temporal_patch_size, k = cfg_.patch_size;
  mx::array x = mx::reshape(pixel_values, {P, c, t, k, k});
  x = mx::transpose(x, {0, 2, 3, 4, 1});  // (P, temporal, patch, patch, in_ch)
  x = mx::reshape(x, {P, -1});

  const mx::array& w = w_.at("visual.patch_embed.proj.weight");
  mx::array w2 = mx::reshape(w, {cfg_.hidden, -1});  // (vit_hidden, patch_flat)
  x = mx::astype(x, w2.dtype());  // compute in the weight's dtype (fp16)
  mx::array out = mx::matmul(x, mx::transpose(w2));  // (num_patches, vit_hidden)
  return mx::add(out, w_.at("visual.patch_embed.proj.bias"));
}

}  // namespace mlxforge
