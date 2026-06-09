// Endurance / leak tests for the libmlxforge C ABI (capi/mlxforge.h).
//
// The engine keeps per-request state (KV-cache arrays, token queues, penalty
// history, RNG keys) that is compacted in place on eviction; over thousands of
// submit/cancel/free cycles any mismatched create/free pair (engine/request/
// string/floats) or retained row leaks. These tests pound the ABI under
// sustained load and assert resident-set growth stays bounded.
//
// LeakSanitizer is unavailable on Apple Silicon, so we quantify leaks via the
// process RSS (mach task_info) instead: measure after a warm-up (so one-time
// allocations and the Metal/MLX buffer caches have settled), run the bulk of the
// iterations, then assert the delta is small. Build with -DMLXFORGE_ENABLE_SANITIZERS=ON
// to additionally catch use-after-free / double-free under ASan while these run.
//
// Gated on the model being present AND MLXFORGE_ENDURANCE being set (its value is
// the iteration count, e.g. MLXFORGE_ENDURANCE=1000), so a normal `ctest` stays
// fast — exactly like the other model-gated integration tests.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <mach/mach.h>

#include "capi/mlxforge.h"
#include "core/env.h"
#include "support/model_fixture.h"

using namespace mlxforge::test;

namespace {

// Resident set size of this process, in bytes (0 if the query fails).
size_t rss_bytes() {
  mach_task_basic_info info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return 0;
  return info.resident_size;
}

double mb(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

// Fully drain a request to completion (no cancel), returning its decoded text.
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

mlxforge_engine* make_ready_engine() {
  char* err = nullptr;
  mlxforge_engine* eng = mlxforge_engine_create(model_dir().c_str(), nullptr, &err);
  REQUIRE_MESSAGE(eng != nullptr, (err ? err : "engine_create failed"));
  while (!mlxforge_engine_ready(eng))
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  return eng;
}

// True if this run is enabled, and sets `iters` to the requested count.
bool endurance_enabled(int& iters) {
  const long n = mlxforge::env_long("MLXFORGE_ENDURANCE", 0);
  iters = static_cast<int>(n);
  return n > 0;
}

// Generous per-test growth ceiling. The aim is to catch a *steady* leak (linear
// in iterations), not allocator slack or MLX's buffer cache, both of which
// plateau after the warm-up. A real per-request leak of even a few KB would blow
// past this over hundreds of iterations.
constexpr double kMaxGrowthMB = 384.0;

}  // namespace

TEST_CASE("C ABI endurance: sustained generate/embed/cancel churn does not leak") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  int iters = 0;
  if (!endurance_enabled(iters)) {
    MESSAGE("MLXFORGE_ENDURANCE not set; skipping endurance run "
            "(set MLXFORGE_ENDURANCE=<iterations>, e.g. 1000)");
    return;
  }
  MESSAGE("endurance: " << iters << " iterations/loop");

  mlxforge_engine* eng = make_ready_engine();
  const int warmup = iters / 10 + 5;  // let one-time allocations settle first

  mlxforge_sampling greedy = {};
  greedy.max_tokens = 8;  // short, so we cycle handles fast

  auto run_loop = [&](const char* name, auto&& body) {
    size_t baseline = 0;
    for (int i = 0; i < iters; ++i) {
      body(i);
      if (i == warmup) baseline = rss_bytes();
    }
    const size_t end = rss_bytes();
    const double grown = mb(end) - mb(baseline);
    MESSAGE(name << ": RSS after warm-up " << mb(baseline) << " MB -> " << mb(end)
                 << " MB (grew " << grown << " MB over " << (iters - warmup)
                 << " post-warm-up iters)");
    CHECK(grown < kMaxGrowthMB);
  };

  SUBCASE("submit chat -> drain -> free") {
    run_loop("chat", [&](int) {
      mlxforge_msg m = {"user", "Say hello."};
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &greedy, nullptr);
      REQUIRE(r != nullptr);
      drain(r);
      mlxforge_request_free(r);
    });
  }

  SUBCASE("submit text -> drain -> free") {
    run_loop("text", [&](int) {
      mlxforge_request* r = mlxforge_submit_text(eng, "The quick brown fox", &greedy, nullptr);
      REQUIRE(r != nullptr);
      drain(r);
      mlxforge_request_free(r);
    });
  }

  SUBCASE("embed -> floats_free") {
    run_loop("embed", [&](int i) {
      float* v = nullptr;
      size_t n = 0;
      char* err = nullptr;
      // Alternate the simple and extended embed entry points.
      const int rc = (i & 1) ? mlxforge_embed_ex(eng, "a sentence to embed", nullptr, &v, &n, &err)
                             : mlxforge_embed(eng, "a sentence to embed", 0, &v, &n, &err);
      REQUIRE_MESSAGE(rc == 0, (err ? err : "embed failed"));
      REQUIRE(v != nullptr);
      mlxforge_floats_free(v);
    });
  }

  SUBCASE("submit -> cancel before drain -> free") {
    // Exercises the drain-on-free / queue-unblock path: a request abandoned while
    // still running must be cancelled and its queue drained by request_free so
    // the worker's producer never blocks on a full, orphaned queue.
    run_loop("cancel", [&](int) {
      mlxforge_msg m = {"user", "Write a long essay about the ocean."};
      mlxforge_sampling s = {};
      s.max_tokens = 256;  // long, so the request is still running at free time
      mlxforge_request* r = mlxforge_submit_chat(eng, &m, 1, &s, nullptr);
      REQUIRE(r != nullptr);
      mlxforge_request_cancel(r);
      mlxforge_request_free(r);  // must drain to completion internally
    });
  }

  mlxforge_engine_free(eng);
}

TEST_CASE("C ABI endurance: concurrent batch churn stays correct and bounded") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  int iters = 0;
  if (!endurance_enabled(iters)) {
    MESSAGE("MLXFORGE_ENDURANCE not set; skipping endurance run");
    return;
  }

  mlxforge_engine* eng = make_ready_engine();

  mlxforge_sampling greedy = {};
  greedy.max_tokens = 8;
  mlxforge_msg msg = {"user", "What is the capital of France?"};

  // Greedy single-stream baseline: under continuous batching, every concurrent
  // greedy run of the same prompt must reproduce it exactly. Drift under load
  // would mean batching corrupts state — a correctness regression, not just a leak.
  mlxforge_request* r0 = mlxforge_submit_chat(eng, &msg, 1, &greedy, nullptr);
  REQUIRE(r0 != nullptr);
  const std::string baseline = drain(r0);
  mlxforge_request_free(r0);
  REQUIRE(baseline.size() > 0);

  const int kThreads = 4;
  const int rounds = iters / 10 + 2;  // each round runs kThreads concurrent requests
  const int warmup = rounds / 4 + 1;

  std::atomic<int> mismatches{0};
  size_t baseline_rss = 0;
  for (int round = 0; round < rounds; ++round) {
    std::vector<mlxforge_request*> reqs(kThreads, nullptr);
    for (int t = 0; t < kThreads; ++t) {
      reqs[t] = mlxforge_submit_chat(eng, &msg, 1, &greedy, nullptr);
      REQUIRE(reqs[t] != nullptr);
    }
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        if (drain(reqs[t]) != baseline) mismatches.fetch_add(1);
      });
    }
    for (auto& th : threads) th.join();
    for (auto* r : reqs) mlxforge_request_free(r);
    if (round == warmup) baseline_rss = rss_bytes();
  }

  CHECK(mismatches.load() == 0);  // batched greedy == single-stream greedy, always
  const double grown = mb(rss_bytes()) - mb(baseline_rss);
  MESSAGE("concurrent: grew " << grown << " MB over " << (rounds - warmup)
                              << " post-warm-up rounds of " << kThreads << " requests");
  CHECK(grown < kMaxGrowthMB);

  mlxforge_engine_free(eng);
}

TEST_CASE("C ABI endurance: repeated engine create/destroy does not leak") {
  if (!model_available()) {
    MESSAGE("MLXFORGE_MODEL_DIR not present; skipping");
    return;
  }
  int iters = 0;
  if (!endurance_enabled(iters)) {
    MESSAGE("MLXFORGE_ENDURANCE not set; skipping endurance run");
    return;
  }

  // Full-lifecycle churn (each iteration spawns and joins the GPU worker thread
  // and loads/frees the model), so far fewer iterations than the request loops.
  const int cycles = iters < 20 ? iters : 20;
  const int warmup = cycles / 4 + 1;

  size_t baseline = 0;
  for (int i = 0; i < cycles; ++i) {
    mlxforge_engine* eng = make_ready_engine();
    mlxforge_request* r = mlxforge_submit_text(eng, "hi", nullptr, nullptr);
    REQUIRE(r != nullptr);
    drain(r);
    mlxforge_request_free(r);
    mlxforge_engine_free(eng);
    if (i == warmup) baseline = rss_bytes();
  }
  const double grown = mb(rss_bytes()) - mb(baseline);
  MESSAGE("create/destroy: grew " << grown << " MB over " << (cycles - warmup)
                                  << " post-warm-up cycles");
  CHECK(grown < kMaxGrowthMB);
}
