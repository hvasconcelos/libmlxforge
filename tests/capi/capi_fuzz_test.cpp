// Fuzz / robustness tests for the libmlxforge C ABI (capi/mlxforge.h).
//
// The ABI's whole contract is "no C++ exception ever crosses the boundary; bad
// input comes back as an error, not a crash." capi_test.cpp proves the happy
// path; this file proves the hostile one: NULL/degenerate arguments, malformed
// model files, and (on a live engine) adversarial sampling / prompt / schema
// inputs. Every case asserts the same thing — the library returns an error or a
// valid result and never crashes, hangs, or leaks on the failure path.
//
// Inputs are generated from a fixed-seed PRNG so a failure reproduces exactly on
// the next run (CLAUDE.md bans wall-clock/Math.random in tests for this reason).
//
// The adversarial-sampling subcase includes inputs that used to abort the process
// (top_k > vocab reached mx::topk, which throws, and the worker re-threw out of
// its thread -> std::terminate; likewise a non-finite temperature). The library
// now sanitizes these (sampler clamps top_k to vocab; the C ABI's to_params
// guards non-finite values), so this file is the regression gate that keeps them
// crash-free.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "capi/mlxforge.h"
#include "core/model_source.h"  // looks_like_repo_id — keep fuzz specs offline
#include "support/model_fixture.h"

using namespace mlxforge::test;
namespace fs = std::filesystem;

namespace {

// Tiny deterministic xorshift64* PRNG: a fixed seed makes every fuzz iteration
// reproducible, so a failing case is the same case next run.
struct Fuzzer {
  uint64_t s;
  explicit Fuzzer(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1Dull;
  }
  int in(int lo, int hi) { return lo + static_cast<int>(next() % uint64_t(hi - lo + 1)); }
  // `n` arbitrary non-NUL bytes (the ABI takes NUL-terminated strings, so an
  // embedded NUL would just truncate; that is its own, separate concern).
  std::string bytes(int n) {
    std::string out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(static_cast<char>(in(1, 255)));
    return out;
  }
};

// True if resolve_model_dir would try a *network* download for this spec (a bare
// repo id, or an "org/name:VARIANT" GGUF fetch). The offline fuzz must never feed
// engine_create such a spec — it would hang on the network — so we skip them; the
// point here is malformed-input safety, not the resolver.
bool would_hit_network(const std::string& spec) {
  if (mlxforge::looks_like_repo_id(spec)) return true;
  const auto colon = spec.find(':');
  return colon != std::string::npos && mlxforge::looks_like_repo_id(spec.substr(0, colon));
}

// On a failure path, free the message and assert it was non-empty (a NULL err is
// allowed: the caller may opt out of the message).
void expect_err_and_free(char* err) {
  if (err) {
    CHECK(std::strlen(err) > 0);
    mlxforge_string_free(err);
  }
}

}  // namespace

TEST_CASE("C ABI NULL-safety: every entry point survives NULL handles") {
  // The free-family and cancel are documented to ignore NULL.
  mlxforge_string_free(nullptr);
  mlxforge_floats_free(nullptr);
  mlxforge_engine_free(nullptr);
  mlxforge_request_free(nullptr);
  mlxforge_request_cancel(nullptr);

  // Accessors on NULL return the documented empty/zero, not a crash.
  CHECK(mlxforge_engine_ready(nullptr) == 0);
  CHECK(std::string(mlxforge_engine_model_name(nullptr)).empty());
  CHECK(std::string(mlxforge_request_finish_reason(nullptr)).empty());

  // request_next(NULL) is a terminal error and must null its out-param even when
  // it is passed a non-NULL (poisoned) pointer.
  char* text = reinterpret_cast<char*>(0x1);
  CHECK(mlxforge_request_next(nullptr, &text) == -1);
  CHECK(text == nullptr);
  CHECK(mlxforge_request_next(nullptr, nullptr) == -1);  // NULL out-param too

  // engine_create with a null/empty spec => error message, no engine.
  char* err = nullptr;
  CHECK(mlxforge_engine_create(nullptr, nullptr, &err) == nullptr);
  expect_err_and_free(err);
  err = nullptr;
  CHECK(mlxforge_engine_create("", nullptr, &err) == nullptr);
  expect_err_and_free(err);
  CHECK(mlxforge_engine_create(nullptr, nullptr, nullptr) == nullptr);  // NULL err ok

  // submit/embed on a NULL engine => error/non-zero return, out-params nulled.
  err = nullptr;
  CHECK(mlxforge_submit_text(nullptr, "hi", nullptr, &err) == nullptr);
  expect_err_and_free(err);
  err = nullptr;
  mlxforge_msg m = {"user", "hi"};
  CHECK(mlxforge_submit_chat(nullptr, &m, 1, nullptr, &err) == nullptr);
  expect_err_and_free(err);
  err = nullptr;
  float* v = reinterpret_cast<float*>(0x1);
  size_t n = 123;
  CHECK(mlxforge_embed(nullptr, "hi", 0, &v, &n, &err) != 0);
  CHECK(v == nullptr);
  CHECK(n == 0);
  expect_err_and_free(err);
  err = nullptr;
  v = reinterpret_cast<float*>(0x1);
  CHECK(mlxforge_embed_ex(nullptr, "hi", nullptr, &v, &n, &err) != 0);
  CHECK(v == nullptr);
  expect_err_and_free(err);
}

TEST_CASE("C ABI engine_create rejects malformed specs as errors, never crashes") {
  Fuzzer f(0xC0FFEE);
  // Pathological-but-offline specs (none is a loadable model, none triggers a
  // network resolve) plus random byte strings.
  std::vector<std::string> specs = {
      " ",        "   \t\n", "::::",
      "not-a-real-path",      "/nonexistent/path/to/model",
      "./still/missing",      "a/b/c/d",  // too many slashes => not a repo id
      "bad:variant",                       // colon, but repo half is not a repo id
      "/tmp/does-not-exist.gguf",          // .gguf suffix, file absent
      std::string(8192, 'x'),              // very long, no slash
  };
  for (int i = 0; i < 200; ++i) specs.push_back(f.bytes(f.in(1, 64)));

  for (const std::string& spec : specs) {
    if (spec.empty() || would_hit_network(spec)) continue;  // stay offline
    char* err = nullptr;
    mlxforge_engine* e = mlxforge_engine_create(spec.c_str(), nullptr, &err);
    CHECK(e == nullptr);     // none of these is a loadable model
    expect_err_and_free(err);
    mlxforge_engine_free(e);  // NULL-safe even on the failure path
  }
}

TEST_CASE("C ABI surfaces malformed model files as errors, never crashes") {
  const fs::path base = fs::temp_directory_path() / "mlxforge_fuzz_models";
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base, ec);

  auto try_spec = [](const std::string& spec) {
    char* err = nullptr;
    mlxforge_engine* e = mlxforge_engine_create(spec.c_str(), nullptr, &err);
    CHECK(e == nullptr);
    expect_err_and_free(err);
    mlxforge_engine_free(e);
  };

  // Malformed config.json in a dir: is_model_dir() sees config.json and routes
  // here, so ModelConfig::from_file / the tokenizer load run under the boundary.
  const std::vector<std::string> bad_configs = {
      "",                                       // empty
      "not json at all",                        // garbage
      "{",                                      // truncated
      "{}",                                     // valid json, every field missing
      R"({"model_type":"llama"})",              // missing dims
      R"({"hidden_size":-1,"num_hidden_layers":0})",
      R"({"vocab_size":999999999999999999})",   // absurd
      std::string("{\"x\":\"") + std::string(1 << 16, 'A') + "\"}",  // huge
  };
  int idx = 0;
  for (const auto& cfg : bad_configs) {
    const fs::path dir = base / ("cfg_" + std::to_string(idx++));
    fs::create_directories(dir, ec);
    std::ofstream(dir / "config.json") << cfg;
    try_spec(dir.string());
  }

  // A dir that exists but has no config.json at all (a common user mistake).
  const fs::path empty_dir = base / "empty";
  fs::create_directories(empty_dir, ec);
  try_spec(empty_dir.string());

  // Malformed GGUF files: the custom reader must reject bad magic / truncation.
  Fuzzer f(0xBADF00D);
  for (int i = 0; i < 16; ++i) {
    const fs::path p = base / ("bad_" + std::to_string(i) + ".gguf");
    const std::string blob = f.bytes(f.in(0, 128));
    std::ofstream(p, std::ios::binary).write(blob.data(), static_cast<std::streamsize>(blob.size()));
    try_spec(p.string());
  }
  // A file with the right magic but nothing after it (truncated header).
  {
    const fs::path p = base / "magic_only.gguf";
    const char magic[4] = {'G', 'G', 'U', 'F'};
    std::ofstream(p, std::ios::binary).write(magic, 4);
    try_spec(p.string());
  }

  fs::remove_all(base, ec);
}

// ---- Live-engine fuzz (needs the model) ------------------------------------

namespace {

// Drain a request but never run away: cancel after a few chunks and a small cap,
// so even a large/garbage token budget can't hang the suite. Reaching the end
// (the stream terminates and the handle frees cleanly) is the assertion — that
// the engine survived the input. rc is 1 (done) or -1 (clean error); both are
// fine, only a crash/hang would fail.
void drain_bounded(mlxforge_request* r) {
  REQUIRE(r != nullptr);
  char* text = nullptr;
  int chunks = 0;
  int rc;
  while ((rc = mlxforge_request_next(r, &text)) == 0) {
    if (text) {
      mlxforge_string_free(text);
      text = nullptr;
    }
    if (++chunks >= 8) mlxforge_request_cancel(r);  // bound runtime
  }
  CHECK(rc != 0);  // the loop exited => the stream terminated, no hang
  (void)mlxforge_request_finish_reason(r);  // must be callable, never crashes
  mlxforge_request_free(r);
}

// A ready engine on the small Llama model, shared by the live-fuzz subcases.
mlxforge_engine* make_ready_engine() {
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  return eng;
}

}  // namespace

TEST_CASE("C ABI tolerates adversarial request inputs on a live engine") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  mlxforge_engine* eng = make_ready_engine();

  SUBCASE("out-of-range and non-finite sampling parameters") {
    // These exercise the ABI's input sanitization in to_params (disabled
    // sentinels, non-finite guards) and the sampler's top_k-vs-vocab clamp.
    // Includes the formerly-crashing vectors (top_k >> vocab, NaN/Inf
    // temperature, min_p > 1): they must now decode cleanly, not abort.
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<mlxforge_sampling> set;
    auto add = [&](float temp, int tk, float tp, float mp, float rp, float fp, float pp) {
      mlxforge_sampling s = {};
      s.max_tokens = 6;
      s.temperature = temp;
      s.top_k = tk;
      s.top_p = tp;
      s.min_p = mp;
      s.repetition_penalty = rp;
      s.frequency_penalty = fp;
      s.presence_penalty = pp;
      set.push_back(s);
    };
    add(0.0f, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);           // all-zero => greedy
    add(0.8f, std::numeric_limits<int>::min(), 0, 0, 0, 0, 0);  // top_k clamps to 0
    add(0.8f, -1, -5.0f, 0, 0, 0, 0);                     // negatives => disabled
    add(0.8f, 0, 5.0f, 0, 0, 0, 0);                       // top_p > 1 => disabled
    add(0.8f, 50, 0.9f, 0.05f, 0, 0, 0);                  // a valid stochastic combo
    add(0.8f, 0, 0, -1.0f, 0, 0, 0);                      // min_p < 0 => disabled
    add(0.8f, 0, 0, 0, -1.0f, 0, 0);                      // rep penalty <= 0 => disabled
    add(0.8f, 0, 0, 0, 2.0f, 1000.0f, -1000.0f);          // large finite penalties
    add(-1e30f, 0, 0, 0, 0, 0, 0);                        // very negative => greedy
    add(1e30f, 0, 0, 0, 0, 0, 0);                         // very large temp
    add(0.8f, 1 << 30, 0, 0, 0, 0, 0);                    // top_k >> vocab => clamped
    add(nan, 0, 0, 0, 0, 0, 0);                           // NaN temp => greedy
    add(inf, 0, 0, 0, 0, 0, 0);                           // +Inf temp => greedy
    add(-inf, 0, 0, 0, 0, 0, 0);                          // -Inf temp => greedy
    add(0.8f, 0, 0, 2.0f, 0, 0, 0);                       // min_p > 1 => disabled
    add(0.8f, 0, 0, 0, inf, nan, inf);                    // non-finite penalties sanitized
    for (auto& s : set) {
      char* err = nullptr;
      mlxforge_msg m = {"user", "Tell me a short story."};
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &s, &err);
      if (!r) {
        expect_err_and_free(err);
        continue;  // a clean rejection is acceptable
      }
      drain_bounded(r);
    }
    // A huge token budget must be bounded by cancellation, not run forever.
    {
      mlxforge_sampling s = {};
      s.max_tokens = 1 << 28;
      mlxforge_msg m = {"user", "Count."};
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &s, nullptr);
      drain_bounded(r);
    }
    // Negative/zero max_tokens falls back to the default budget (then bounded).
    for (int mt : {std::numeric_limits<int>::min(), -1, 0, 1}) {
      mlxforge_sampling s = {};
      s.max_tokens = mt;
      mlxforge_request* r = mlxforge_submit_text(eng, "hello", &s, nullptr);
      drain_bounded(r);
    }
  }

  SUBCASE("adversarial prompts, roles, and message shapes") {
    Fuzzer f(0x1234);
    mlxforge_sampling s = {};
    s.max_tokens = 6;

    // Invalid UTF-8 / control bytes as raw text (kept to a few KB so a giant
    // prefill can't OOM the suite; byte-level BPE tokenizes any bytes).
    std::vector<std::string> prompts = {
        "",
        std::string(1, '\x80'),         // lone continuation byte
        "\xff\xfe\xfd",                  // invalid lead bytes
        "\xed\xa0\x80",                  // UTF-16 surrogate encoded as UTF-8
        std::string("\x01\x02\x03 control chars"),
        std::string(4096, 'a'),         // moderately long
    };
    for (int i = 0; i < 32; ++i) prompts.push_back(f.bytes(f.in(0, 256)));
    for (const auto& p : prompts) {
      mlxforge_request* r = mlxforge_submit_text(eng, p.c_str(), &s, nullptr);
      if (r) drain_bounded(r);
    }

    // Zero messages, bogus roles, NULL fields, large message arrays.
    {
      char* err = nullptr;
      mlxforge_request* r = mlxforge_submit_chat(eng, nullptr, 0, &s, &err);
      if (r) drain_bounded(r);
      else expect_err_and_free(err);
    }
    {
      std::vector<mlxforge_msg> many;
      for (int i = 0; i < 64; ++i) many.push_back({"banana", "x"});  // unknown role
      char* err = nullptr;
      mlxforge_request* r =
          mlxforge_submit_chat(eng, many.data(), many.size(), &s, &err);
      if (r) drain_bounded(r);
      else expect_err_and_free(err);
    }
    {
      mlxforge_msg m = {nullptr, nullptr};  // ABI defaults role->"user", content->""
      char* err = nullptr;
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &s, &err);
      if (r) drain_bounded(r);
      else expect_err_and_free(err);
    }
  }

  SUBCASE("adversarial json_schema strings") {
    // from_schema_string is lenient (allow_exceptions=false; unsupported => free
    // JSON), so none of these should error — but all must stay crash-free.
    std::vector<std::string> schemas = {
        "json",
        "JSON",
        "not json",
        "{",
        "{}",
        R"({"type":"banana"})",
        R"({"type":"object","properties":{"a":{"type":"nope"}}})",
        R"({"type":"object","properties":{}})",
        std::string("{") + std::string(1 << 16, ' ') + "}",   // huge whitespace
        std::string(R"({"type":"object","properties":)") + std::string(1000, '['),
    };
    for (const auto& sc : schemas) {
      mlxforge_sampling s = {};
      s.max_tokens = 8;
      s.json_schema = sc.c_str();
      char* err = nullptr;
      mlxforge_msg m = {"user", "Give me JSON."};
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &s, &err);
      if (r) drain_bounded(r);
      else expect_err_and_free(err);
    }
  }

  SUBCASE("adversarial embedding inputs") {
    Fuzzer f(0x5EED);
    // Pooling / add_eos out of their {-1,0,1} range, skip_normalize non-bool,
    // odd instructions, weird bytes. Embedding errors are caught inside the
    // worker (handle_embedding) and surfaced as rc!=0, so this is fully safe.
    std::vector<std::string> texts = {"", "hello", "\xff\xfe", std::string(2048, 'z')};
    for (int i = 0; i < 16; ++i) texts.push_back(f.bytes(f.in(0, 128)));
    for (const auto& t : texts) {
      mlxforge_embed_opts o = {};
      o.pooling = f.in(-3, 3);
      o.add_eos = f.in(-3, 3);
      o.skip_normalize = f.in(-1, 2);
      const std::string instr = ((f.next() & 1u) == 0u) ? std::string() : f.bytes(f.in(0, 64));
      o.instruction = instr.empty() ? nullptr : instr.c_str();
      float* v = nullptr;
      size_t n = 0;
      char* err = nullptr;
      int rc = mlxforge_embed_ex(eng, t.c_str(), &o, &v, &n, &err);
      if (rc == 0) {
        CHECK(v != nullptr);
        CHECK(n > 0);
        mlxforge_floats_free(v);
      } else {
        CHECK(v == nullptr);
        expect_err_and_free(err);
      }
    }
  }

  mlxforge_engine_free(eng);
}
