// Mistral-7B-Instruct-v0.3 support: the LlamaModel forward pass is architecture-
// shared, so the proof it also serves Mistral is a golden-reference check against
// mlx-lm (reference/fixtures_mistral, dumped via `dump_ref.py --model mistral`).
// Self-skips when the 4-bit Mistral model is not in the local HF cache.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "core/config.h"
#include "runtime/single_stream.h"
#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/tokenizer.h"

#include "mlx/ops.h"

using namespace mlxforge::test;

namespace {
std::string fixdir() { return MLXFORGE_MISTRAL_REF_FIXTURES_DIR; }
}  // namespace

TEST_CASE("Mistral: forward-pass intermediates + greedy stream match the mlx-lm reference") {
  if (!mistral_available()) {
    MESSAGE("MLXFORGE_MISTRAL_MODEL_DIR not present; skipping golden-reference check");
    return;
  }
  mlxforge::LlamaModel& model = shared_mistral_model();

  std::vector<int> ids = load_token_ids_at(fixdir(), "prompt_0_ids.npy");
  const int T = static_cast<int>(ids.size());
  mx::array tokens(ids.data(), {1, T}, mx::int32);

  // Embedding lookup and a full decoder block (layer 0).
  mx::array h = model.embed(tokens);
  assert_close(h, load_npy_at(fixdir(), "embeddings.npy"));
  assert_close(model.decoder_block(h, /*layer=*/0), load_npy_at(fixdir(), "block0.npy"));

  // Full forward to last-position logits + first-token argmax.
  mx::array logits = model.forward(tokens);  // (1, T, vocab)
  const int vocab = logits.shape()[2];
  mx::array last = mx::reshape(mx::slice(logits, {0, T - 1, 0}, {1, T, vocab}), {1, vocab});
  assert_close(last, load_npy_at(fixdir(), "logits_last.npy"));

  std::vector<int> argmax = load_token_ids_at(fixdir(), "argmax.npy");
  mx::array got = mx::astype(mx::argmax(last, /*axis=*/-1), mx::int32);
  mx::eval(got);
  CHECK(got.data<int32_t>()[0] == argmax[0]);

  // Greedy continuation reproduces the full-recompute reference oracle exactly.
  std::vector<int> expected_greedy = load_token_ids_at(fixdir(), "greedy_tokens.npy");
  mlxforge::GenerateResult r =
      mlxforge::greedy_generate(model, ids, static_cast<int>(expected_greedy.size()),
                                model.config().eos_token_ids);
  assert_tokens_equal(r.tokens, expected_greedy);
}

TEST_CASE("Mistral: chat template encodes to the mlx-lm [INST] token stream") {
  if (!mistral_available()) {
    MESSAGE("MLXFORGE_MISTRAL_MODEL_DIR not present; skipping chat parity check");
    return;
  }
  const mlxforge::ModelConfig& cfg = shared_mistral_model().config();
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      mistral_model_dir() + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type));

  std::vector<mlxforge::Tokenizer::Message> messages = {
      {"user", "What is the capital of France?"}};
  CHECK(tok.apply_chat_template(messages, /*add_generation_prompt=*/true) ==
        load_token_ids_at(fixdir(), "chat_ids.npy"));
}
