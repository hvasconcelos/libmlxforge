// Engine: core inference orchestrator — an embeddable object, not dependent on HTTP.
// Handles the end-to-end model startup: resolves the user model spec (local dir, repo id, or .gguf),
// loads config and tokenizer (GGUF or safetensors), and boots a Worker (on its own thread)
// that interacts with the GPU. Sits above the Scheduler, which handles request submission.
// The Engine exposes only minimal access points for a consumer: providing config, tokenizer,
// the request submission seam, and status/metrics. Used both in HTTP servers and embedded apps.
//
// Usage pattern: create an Engine, build a Request, submit via scheduler().submit(req),
// then drain output tokens from req->tokens. The Engine owns its Worker for its lifetime;
// only that thread directly calls into MLX, preserving thread-safety.
#pragma once

#include <string>
#include <vector>

#include "core/config.h"
#include "runtime/metrics.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "tokenizer/tokenizer.h"

namespace mlxforge {

// Lightweight configuration for engine construction.
struct EngineConfig {
  std::string model_spec;  // Model description: local directory, HuggingFace repo id, or .gguf file path (to be resolved internally)
  int max_waiting = 256;   // Maximum length of the Scheduler's waiting queue; 0 disables the cap
  // KV-cache quantization (engine-wide: the batched cache's storage is shared
  // across rows, so this cannot be per-request). 0 = dense fp16 (default);
  // 8 or 4 store the cache as mx::quantize triplets, matching mlx-lm's
  // QuantizedKVCache numerics. Validated against the model at construction:
  // unsupported combinations (vision/hybrid models, group_size not dividing
  // head_dim) throw rather than silently falling back.
  int kv_bits = 0;
  int kv_group_size = 64;
  // Prefix cache (engine-wide, like kv_bits: the pool stores the engine's
  // single storage layout). Off by default. When on, finished rows' KV is
  // harvested into a block pool and later prompts sharing a token prefix skip
  // that part of prefill. Validated at construction: hybrid (Qwen3.5) and
  // vision-language models are rejected (no golden gate for those paths yet).
  bool prefix_cache = false;
  int kv_block_size = 256;                  // pool granularity, power of two in [16, 4096]
  std::size_t kv_pool_bytes = 1ull << 30;   // pooled-KV RAM budget; 0 = unbounded
  // SSD spill tier (requires prefix_cache): RAM-evicted blocks persist under
  // this directory and survive engine restarts. Empty = no spill.
  std::string kv_spill_dir;
  std::size_t kv_spill_bytes = 0;           // disk budget; 0 = unbounded
  // Chunked-prefill interleaving: admissions prefill `prefill_chunk` tokens per
  // worker iteration with a decode step in between, so in-flight rows keep
  // streaming during long or queued prefills. On by default (256 tokens —
  // benchmarked sweet spot); 0 = monolithic prefill per admission. Negative
  // values are rejected at construction.
  int prefill_chunk = 256;
  // Multi-row GEMV decode kernels (model/skinny_matmul): dense fp16 matmuls of
  // the batched-decode shape (B in [2, 16], L == 1) bypass MLX's tiled GEMM,
  // which runs at a fraction of GEMV bandwidth there (ml-explore/mlx#3661).
  // On by default; logits may differ from the stock kernel at fp16-noise scale
  // (fp32 accumulation in a different order), token-equality gated in tests.
  bool skinny_mm = true;
  // RoPE-scaling override (vLLM --rope-scaling style): a JSON object replacing
  // the checkpoint's rope_scaling config, e.g.
  //   {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}
  // Empty = use the checkpoint's config as-is. Like kv_bits, unsupported setups
  // (unknown rope_type, hybrid/vision models, GGUF) FAIL engine creation —
  // never a silent fall-back to unscaled RoPE.
  std::string rope_scaling;
};

// Per-call embedding options. The two int fields are tri-state: -1 means "use
// the model's detected default" (e.g. a Qwen3-Embedding checkpoint defaults to
// last-token pooling + a trailing EOS), so a bare embed(text) just works.
struct EmbedOptions {
  int pooling = -1;          // -1 = detected default; 0 = mean; 1 = last token
  int add_eos = -1;          // -1 = detected default; 0 = off; 1 = append the model's EOS id
  bool normalize = true;     // L2-normalize the pooled vector (cosine == dot product)
  std::string instruction;   // optional; wraps text as "Instruct: {instruction}\nQuery: {text}"
};

class Engine {
 public:
  // Constructor: resolves the model spec, loads config/tokenizer on the caller's thread,
  // then spawns a Worker which loads weights and must be run on its own thread due to MLX constraints.
  // Throws if the spec is invalid or there is a loading failure.
  explicit Engine(EngineConfig cfg);
  ~Engine();  // Ensures all in-flight requests drain and the Worker thread stops cleanly

  // Disable copying/moving: Engine must remain unique, as it manages thread-bound objects.
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Main submission entrypoint: get a reference to the Scheduler for Request lifecycle/queueing.
  // Non-const: users will submit() Requests, mutating scheduler state.
  Scheduler& scheduler() { return scheduler_; }

  // Access the loaded Tokenizer (thread-safe constant interface)
  const Tokenizer& tokenizer() const { return tok_; }
  // Access the loaded ModelConfig (i.e. model hyperparameters, architecture, etc)
  const ModelConfig& config() const { return cfg_; }
  // Access the resolved model name (as supplied by the user, useful for HTTP responses etc.)
  const std::string& model_name() const { return model_name_; }

  // Query whether the model is loaded and ready for inference (Worker finished initialization)
  bool ready() const { return worker_.ready(); }
  // Fetch running metrics (queues, timing, active requests, etc.) from the Worker
  WorkerMetrics metrics() const { return worker_.metrics(); }

  // Embed `text` into a (by default unit-normalized) vector (synchronous):
  // applies any instruction wrap, encodes, appends EOS when requested, submits a
  // one-shot embedding request through the scheduler, and blocks for the result.
  // With default `opts` the model's detected defaults apply (a Qwen3-Embedding
  // checkpoint uses last-token pooling + a trailing EOS). Returns {} on failure.
  std::vector<float> embed(const std::string& text, const EmbedOptions& opts = {});

  // Explicitly drain all remaining requests/queues and join/stop the Worker thread.
  // Idempotent: also called by the destructor, so only use if early shutdown/drain is desired.
  void stop() { worker_.stop(); }

 private:
  // Internal struct: stores everything loaded on the caller thread prior to Worker start
  // (no MLX arrays instantiated yet — just config, tokenizer, and resolved model dir).
  struct Loaded {
    std::string dir;           // Fully resolved model/weights directory
    bool is_gguf = false;      // Was the model GGUF format?
    ModelConfig config;        // Parsed model config
    Tokenizer tokenizer;       // Parsed tokenizer
    // Embedding defaults sniffed from the model dir's sentence-transformers
    // sidecar (1_Pooling/config.json), used when EmbedOptions leaves a field at
    // -1. Plain LLMs keep mean pooling / no EOS; Qwen3-Embedding flips to last.
    int embed_pooling_default = 0;     // mlxforge::Pooling (0 = mean, 1 = last)
    bool embed_add_eos_default = false;
  };

  // Resolve model spec path, parse config/tokenizer, etc. Applies the optional
  // rope-scaling override and validates it — this is the caller-thread rejection
  // point for unsupported rope configs (the worker thread cannot throw cleanly).
  static Loaded load_head(const std::string& spec, const std::string& rope_scaling);
  // Factory builder: creates a Worker::ModelFactory, handling weight loading with proper backend
  static Worker::ModelFactory make_factory(std::string dir, bool is_gguf,
                                           std::string rope_scaling);

  // Private delegating ctor, used internally after head-loading step is complete.
  Engine(EngineConfig cfg, Loaded loaded);

  // --- Engine state (order is important for destruction) ---

  std::string model_name_;    // Echoes user-supplied spec (for e.g. server listing endpoints)
  ModelConfig cfg_;           // Model config, immutable after load
  Tokenizer tok_;             // Tokenizer, immutable after load

  // Detected embedding defaults (see Loaded), consulted by embed() when a caller
  // leaves an EmbedOptions field at -1.
  int embed_pooling_default_ = 0;
  bool embed_add_eos_default_ = false;

  // Order of these two is critical:
  // Scheduler must outlive Worker, because Worker holds a pointer to scheduler_.
  Scheduler scheduler_;
  Worker worker_;
};

}  // namespace mlxforge
