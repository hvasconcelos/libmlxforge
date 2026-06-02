// Validates the from-scratch byte-level BPE tokenizer against committed golden
// ids dumped from mlx-lm (reference/fixtures/tokenizer_corpus.json), over a
// corpus that exercises the pre-tokenizer's edge cases (whitespace runs,
// newlines, contractions, digits, CJK, accented Latin, emoji/ZWJ, code, inline
// special tokens). Model-gated: self-skips when the model isn't present.
#include <doctest/doctest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "support/model_fixture.h"
#include "support/reference.h"
#include "tokenizer/bpe.h"
#include "tokenizer/tokenizer.h"

using namespace mlxforge::test;

namespace {

std::string tokenizer_path() { return std::string(MLXFORGE_MODEL_DIR) + "/tokenizer.json"; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

TEST_CASE("BpeTokenizer matches the mlx-lm golden ids on a diverse corpus") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  // Llama-3.2 is byte-level BPE, so the wrapper uses our own implementation.
  CHECK(mlxforge::BpeTokenizer::is_supported(read_file(tokenizer_path())));

  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(tokenizer_path());

  const std::string corpus_path = std::string(MLXFORGE_REF_FIXTURES_DIR) + "/tokenizer_corpus.json";
  nlohmann::json corpus = nlohmann::json::parse(read_file(corpus_path));
  REQUIRE(corpus.is_array());
  REQUIRE(corpus.size() > 0);

  for (const auto& entry : corpus) {
    const std::string text = entry["text"].get<std::string>();
    const std::vector<int> expected = entry["ids"].get<std::vector<int>>();
    INFO("input: " << text);
    // tok.encode prepends BOS, matching mlx-lm's tok.encode (how the fixture was dumped).
    assert_tokens_equal(tok.encode(text), expected);

    // Decode round-trips for the special-token-free strings (decode skips BOS
    // and other special ids).
    if (text.find("<|") == std::string::npos) {
      CHECK(tok.decode(tok.encode(text)) == text);
    }
  }
}
