// Exercises the engine through ONLY the C ABI (capi/mlxforge.h) — no C++ engine
// types — proving the embeddable product surface works end to end, that errors
// are reported as messages (not exceptions), and that concurrent requests batch
// while preserving greedy determinism (batched greedy == single-stream greedy).
#include <doctest/doctest.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "capi/mlxforge.h"
#include "support/model_fixture.h"

using namespace mlxforge::test;

namespace {

// Drain a request to completion through the C ABI, returning its decoded text.
std::string drain(mlxforge_request* req) {
  std::string out;
  char* text = nullptr;
  while (mlxforge_request_next(req, &text) == 0) {
    if (text) {
      out += text;
      mlxforge_string_free(text);
      text = nullptr;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("C ABI reports its version and ABI level") {
  CHECK(std::string(mlxforge_version()).size() > 0);
  CHECK(mlxforge_abi_version() == MLXFORGE_ABI_VERSION);
}

TEST_CASE("C ABI surfaces a bad spec as an error message, not a crash") {
  char* err = nullptr;
  mlxforge_engine* e = mlxforge_engine_create("", nullptr, &err);
  CHECK(e == nullptr);
  REQUIRE(err != nullptr);
  CHECK(std::string(err).size() > 0);
  mlxforge_string_free(err);
}

TEST_CASE("C ABI generates text and batches concurrent requests deterministically") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }

  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Greedy (zeroed sampling => argmax), deterministic baseline at batch size 1.
  mlxforge_sampling greedy = {};
  greedy.max_tokens = 16;

  mlxforge_msg msg = {"user", "What is the capital of France?"};
  mlxforge_request* r0 = mlxforge_submit_chat(eng, &msg, 1, &greedy, &err);
  REQUIRE_MESSAGE(r0 != nullptr, (err ? err : "submit_chat failed"));
  const std::string baseline = drain(r0);
  CHECK(std::string(mlxforge_request_finish_reason(r0)).size() > 0);  // "stop" | "length"
  mlxforge_request_free(r0);
  CHECK(baseline.size() > 0);

  // Submit several identical greedy requests at once: they share the batched
  // engine, and — being greedy — must reproduce the single-stream baseline
  // exactly. This validates both the ABI and that batching preserves output.
  const int N = 4;
  std::vector<mlxforge_request*> reqs;
  for (int i = 0; i < N; ++i) {
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &greedy, nullptr);
    REQUIRE(r != nullptr);
    reqs.push_back(r);
  }
  std::vector<std::string> outs(N);
  std::vector<std::thread> threads;
  for (int i = 0; i < N; ++i)
    threads.emplace_back([&, i] { outs[i] = drain(reqs[i]); });
  for (auto& t : threads) t.join();
  for (int i = 0; i < N; ++i) {
    CHECK(outs[i] == baseline);  // batched greedy == single-stream greedy
    mlxforge_request_free(reqs[i]);
  }

  mlxforge_engine_free(eng);
}

TEST_CASE("C ABI constrained decoding forces well-formed JSON") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // JSON mode: the output must be valid JSON regardless of the (small) model.
  {
    mlxforge_sampling s = {};
    s.max_tokens = 96;
    s.json_schema = "json";
    mlxforge_msg msg = {"user", "Describe a person as a JSON object."};
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &s, &err);
    REQUIRE_MESSAGE(r != nullptr, (err ? err : "submit failed"));
    const std::string out = drain(r);
    const std::string reason = mlxforge_request_finish_reason(r);
    mlxforge_request_free(r);
    CAPTURE(out);
    // Whether it stops or hits the token budget, the output is a valid JSON
    // prefix; on a clean stop it is a complete, parseable value.
    if (reason == "stop") {
      auto parsed = nlohmann::json::parse(out, nullptr, /*allow_exceptions=*/false);
      CHECK_FALSE(parsed.is_discarded());
    }
  }

  // Schema mode: a top-level object with required typed keys, in order.
  {
    mlxforge_sampling s = {};
    s.max_tokens = 96;
    s.json_schema =
        R"({"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer"}}})";
    mlxforge_msg msg = {"user", "Give facts about Paris."};
    mlxforge_request* r = mlxforge_submit_chat(eng, &msg, 1, &s, &err);
    REQUIRE_MESSAGE(r != nullptr, (err ? err : "submit failed"));
    const std::string out = drain(r);
    const std::string reason = mlxforge_request_finish_reason(r);
    mlxforge_request_free(r);
    CAPTURE(out);
    if (reason == "stop") {
      auto parsed = nlohmann::json::parse(out, nullptr, /*allow_exceptions=*/false);
      REQUIRE_FALSE(parsed.is_discarded());
      CHECK(parsed.is_object());
      CHECK(parsed.contains("city"));
      CHECK(parsed.contains("population"));
      CHECK(parsed["city"].is_string());
      CHECK(parsed["population"].is_number_integer());
    }
  }

  mlxforge_engine_free(eng);
}
