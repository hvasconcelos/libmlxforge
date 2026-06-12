# Changelog

All notable changes to **mlxforge** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-12

This release adds a from-scratch **Qwen3-VL vision-language** model (image → text),
**first-class Qwen3-Embedding**, and a set of throughput and memory features —
**KV-cache quantization**, a **prefix cache with paged KV storage and an SSD spill
tier**, **chunked-prefill interleaving**, and **multi-row decode GEMV/MMA kernels** —
all golden-reference gated. The C ABI grew from **v1 to v9** (append-only; older
bindings keep working).

### Added

- **Qwen3-VL vision-language (image → text).** A from-scratch ViT encoder + image
  merge + interleaved 3D **M-RoPE** + DeepStack, served **prefill-single /
  decode-batched** (like vLLM/omlx): the ViT + 3D-M-RoPE prefill runs single-stream,
  then the prompt's K/V is adopted into a batch-1 `BatchKVCache` and merged into the
  continuous-batching decode pool, so a generated (pure-text) VL token decodes
  alongside ordinary text rows. Every vision stage is golden-gated against `mlx-vlm`
  (`reference/fixtures_qwen3_vl/`), and batched decode is gated equal to the
  single-stream output. Exposed on the OpenAI and Anthropic chat endpoints (image
  content parts, multiple images per turn) and through the C ABI / bindings.
- **First-class Qwen3-Embedding support.** A bare `engine.embed(text)` now
  self-selects the checkpoint's embedding convention: the engine sniffs the
  sentence-transformers pooling sidecar (`1_Pooling/config.json`) and, for a
  `pooling_mode_lasttoken` model (Qwen3-Embedding), defaults to **last-token
  pooling** with a **trailing EOS** (`<|endoftext|>`); plain LLMs keep mean
  pooling. Retrieval queries take an instruction that renders as
  `Instruct: {instruction}\nQuery:{text}` (Qwen's `get_detailed_instruct`). The
  loader also normalizes **backbone-root** checkpoints (`layers.N.*` / `embed_tokens`
  / `norm`, no `model.` prefix or `lm_head`) to the canonical `model.*` form, so the
  official `Qwen/Qwen3-Embedding-0.6B` repo loads directly. Golden-gated against the
  real model's pooled query/document vectors + token ids.
- **KV-cache quantization (`--kv-bits 8|4`, default fp16).** mlx-lm-matching quantized
  triplet storage + a hand-rolled `quantized_sdpa` attention path for ~1.9× / ~3.6×
  less cache memory — including the active continuous-decode batch, which no other MLX
  server quantizes. Golden-gated teacher-forced and margin-gated against `mlx-lm`'s
  `QuantizedKVCache`. Engine-wide; vision/hybrid models are rejected at engine creation
  (no silent fp16 fallback).
- **Prefix cache + paged (block-pool) KV storage with an SSD spill tier.** Immutable
  prompt-K/V blocks keyed by a salted chain hash are pooled in memory and seed a
  batch-1 cache on a prefix hit (warm == cold, token-exact — only prefill-produced K/V
  is harvested). An optional on-disk tier spills/loads blocks across runs. Engine-wide
  opt-in; vision/hybrid models and spill-without-prefix-cache are rejected at engine
  creation.
- **Chunked-prefill interleaving (`prefill_chunk`, default-on).** Long prompts are
  prefilled in chunks interleaved with the decode batch instead of monopolizing the
  worker — **+25–35% batched throughput and up to ~60% lower TTFT under load**.
- **Multi-row decode kernels (`skinny_mm`, default-on).** A multi-row GEMV path plus a
  `simdgroup_matrix` MMA kernel for the big-weight decode matmuls (batch M ≈ 5–32),
  speeding up batched decode over the per-row baseline.
- **Per-token logprobs** — OpenAI-compatible `logprobs` / `top_logprobs` on the
  completion and chat endpoints.
- **Cross-engine benchmark harness** for repeatable throughput comparison against other
  MLX servers.
- **Full multi-turn chat history + `prompt_tokens`** accounting across the server, C
  ABI, and bindings.
- **C ABI v2 → v9** (append-only) — `mlxforge_embed_ex` + `mlxforge_embed_opts`
  (pooling / add_eos / skip_normalize / instruction; `-1` defers to the model default),
  vision/image submission, logprobs, and the kv-quant / prefix-cache / chunked-prefill
  / skinny_mm engine options. Existing v1 entry points are unchanged; the Node, Swift,
  and Rust bindings expose the new surface.
- **Hardened sampling + C ABI fuzz and endurance tests** (#29) — sampling is hardened
  against hostile / degenerate input, with long-running fuzz and endurance coverage of
  the ABI surface.
- **Technical white paper** (LaTeX) documenting the engine design, under `doc/`.

### Fixed

- **SSE stream termination** — the OpenAI streaming endpoint now terminates the event
  stream correctly (`[DONE]` framing) under client disconnect and completion.

[0.2.0]: https://github.com/hvasconcelos/libmlxforge/releases/tag/0.2.0

## [0.1.0] - 2026-06-09

First release of **`libmlxforge`** — an embeddable, continuously batched LLM
inference engine for Apple Silicon, built from scratch in C++ on the Apple MLX
C++ core. It is the only MLX project that is a *complete, batched* engine
(scheduler + continuous batching + own tokenizer/GGUF/chat templates) designed to
be embedded **in-process** and driven from other languages through a stable C ABI.

### The library (the product)

- **Stable C ABI** (`src/capi/mlxforge.h`, ABI v1) — a small, append-only,
  versioned `extern "C"` surface that never throws across the boundary. Create an
  engine from a model spec, submit chat/text requests, and stream tokens:
  `mlxforge_engine_create` / `_ready` / `_model_name`, `mlxforge_submit_chat` /
  `_submit_text`, `mlxforge_request_next` (token streaming) / `_cancel` /
  `_finish_reason`, plus `mlxforge_embed` for embeddings. Runtime version and ABI
  introspection via `mlxforge_version` / `mlxforge_abi_version`.
- **Continuous batching scheduler** — many concurrent requests share one decode
  loop with a single `async_eval` per step over the whole batch, so throughput
  scales with load instead of one stream at a time.
- **Embeddings** with pooling, exposed directly through the C ABI.
- **Structured / constrained output** for reliable JSON and tool-call generation.
- **Language bindings** on top of the C ABI: **Node**, **Swift**, and **Rust**.
- **Lean dylib** — the released artifact builds with the HTTP server and CLI
  harnesses **off** (no httplib/curl), shipping just the engine. Versioned dylib
  (`VERSION 0.1.0`, `SOVERSION 0`) with distribution packaging, an ABI guard, and
  a conformance kit to keep bindings honest against the ABI.

### Models & formats

- **LLaMA-family decoder-only transformers**: Llama-3.2, Mistral, Qwen3 (dense),
  Qwen3 MoE (sparse mixture-of-experts), and Qwen3.5 hybrid (Gated-DeltaNet, text).
- **Weight formats**: HuggingFace safetensors (fp16 / 4-bit, mixed-bit) and
  **GGUF** (Q4_0/Q4_1/Q8_0, Q4_K/Q5_K/Q6_K) with per-weight quantization, plus
  automatic HuggingFace download and hub-cache resolution.
- **Own tokenizers** behind an `EncoderBackend` interface — byte-level BPE
  (Llama/Qwen) and SentencePiece-BPE (Gemma) — byte-validated against mlx-lm
  golden ids. Chat templates for Llama and ChatML (with Qwen `enable_thinking`).
- **Sampling**: temperature, top-p, min-p, and repetition / frequency / presence
  penalties.

### Correctness

- **Golden-reference gated.** The forward pass, KV cache, RoPE (llama3-scaled),
  and sampling are validated against `mlx-lm` `.npy` fixtures committed under
  `reference/`, guarding against the engine's defining failure mode — silent
  numerical garbage rather than a crash.

### Harnesses (dev/QA only)

- An HTTP server (OpenAI- and Anthropic-compatible endpoints, tool/function
  calling) and a CLI exist to exercise and prove engine stability. They are not
  part of the shipped library.

[0.1.0]: https://github.com/hvasconcelos/libmlxforge/releases/tag/0.1.0
