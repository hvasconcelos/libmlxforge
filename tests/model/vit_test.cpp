// Qwen3-VL ViT encoder, gated stage by stage against reference/fixtures_qwen3_vl
// (dumped from mlx-vlm on the same 4bit checkpoint). Self-skips when the model
// snapshot is absent — a green run without it only covers the pure-logic units.
#include <doctest/doctest.h>

#include "mlx/ops.h"

#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge;
using namespace mlxforge::test;

TEST_CASE("Qwen3-VL ViT: patch embedding matches the reference") {
  if (!qwen3_vl_model_available()) {
    MESSAGE("Qwen3-VL model not found in HF cache; skipping ViT patch-embed test");
    return;
  }
  const VitEncoder& vit = shared_qwen3_vl_vit();

  // pixel_values (num_patches, in_ch*temporal*patch*patch) -> patch embeddings.
  mx::array pixel_values = load_qwen3_vl_npy("pixel_values.npy");
  mx::array pe = vit.patch_embed(pixel_values);
  mx::eval(pe);

  assert_close(pe, load_qwen3_vl_npy("vit_patch_embed.npy"));
}
