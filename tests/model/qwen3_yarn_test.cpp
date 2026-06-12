// Qwen3 + yarn golden-reference checks: the same Qwen3-0.6B weights with the
// yarn rope_scaling from the fixtures_qwen3_yarn manifest injected through the
// engine's override path (apply_rope_scaling_override), gated end to end against
// mlx-lm loaded with the identical model_config injection — the yarn-rescaled
// attention front-half (freqs + mscale-on-input), the full-forward argmax, and
// the greedy token stream. Self-skips unless both the Qwen3 model
// (MLXFORGE_MODEL_DIR_QWEN3) and the committed yarn fixtures are present.
#include <doctest/doctest.h>

#include <fstream>
#include <vector>

#include "mlx/ops.h"
#include "runtime/single_stream.h"
#include "support/model_fixture.h"
#include "support/reference.h"

using namespace mlxforge::test;

namespace {
bool yarn_fixtures_present() {
  return std::ifstream(qwen3_yarn_ref_path("manifest.json")).good();
}
bool yarn_ready() { return qwen3_model_available() && yarn_fixtures_present(); }
}  // namespace

TEST_CASE("Qwen3+yarn: rescaled attention front-half matches the reference") {
  if (!yarn_ready()) {
    MESSAGE("Qwen3 model / yarn fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3Model& model = shared_qwen3_yarn_model();

  // The override took: yarn freqs schedule + attention mscale, both vs mlx-lm.
  assert_close(model.rope_freqs(), load_qwen3_yarn_npy("rope_freqs.npy"), /*rtol=*/1e-5f,
               /*atol=*/1e-3f);
  mx::array ref_mscale = load_qwen3_yarn_npy("rope_mscale.npy");
  mx::eval(ref_mscale);
  CHECK(model.rope_mscale() == doctest::Approx(ref_mscale.item<float>()).epsilon(1e-6));

  std::vector<int> ids = load_qwen3_yarn_token_ids("prompt_0_ids.npy");
  mx::array tokens(ids.data(), {1, static_cast<int>(ids.size())}, mx::int32);
  mx::array emb = model.embed(tokens);
  assert_close(emb, load_qwen3_yarn_npy("embeddings.npy"));

  // Post-(q_norm + mscale + yarn-RoPE) Q/K. If the mscale were dropped or the
  // freqs unscaled, q/k diverge from the reference here.
  mlxforge::DecoderModel::QKV qkv = model.attn_qkv(emb, /*layer=*/0);
  assert_close(qkv.q, load_qwen3_yarn_npy("q_rope0.npy"));
  assert_close(qkv.k, load_qwen3_yarn_npy("k_rope0.npy"));
  assert_close(qkv.v, load_qwen3_yarn_npy("v0.npy"));
}

TEST_CASE("Qwen3+yarn: full forward logits + first-token argmax match the reference") {
  if (!yarn_ready()) {
    MESSAGE("Qwen3 model / yarn fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3Model& model = shared_qwen3_yarn_model();
  std::vector<int> ids = load_qwen3_yarn_token_ids("prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  // Same loosened bound as the plain Qwen3 gate (28 layers of fp16 drift, now
  // also mscale^2-scaled); the exact argmax below is the real correctness gate.
  assert_close(last, load_qwen3_yarn_npy("logits_last.npy"), /*rtol=*/3e-2f, /*atol=*/3e-2f);

  std::vector<int> argmax = load_qwen3_yarn_token_ids("argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);
}

TEST_CASE("Qwen3+yarn: greedy stream matches mlx-lm token-for-token") {
  if (!yarn_ready()) {
    MESSAGE("Qwen3 model / yarn fixtures not present; skipping golden-reference check");
    return;
  }
  mlxforge::Qwen3Model& model = shared_qwen3_yarn_model();
  std::vector<int> prompt = load_qwen3_yarn_token_ids("prompt_0_ids.npy");
  std::vector<int> ref = load_qwen3_yarn_token_ids("greedy_tokens.npy");

  mlxforge::GenerateResult r = mlxforge::greedy_generate(
      model, prompt, /*max_tokens=*/static_cast<int>(ref.size()), model.config().eos_token_ids);
  assert_tokens_equal(r.tokens, ref);
}
