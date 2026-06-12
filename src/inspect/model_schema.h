// Architecture schema for the CLI `schematic` command.
//
// Builds a structured, JSON-serializable description of a model — every tensor
// (logical shape, dtype, quantization, bytes), per-component and per-layer
// parameter aggregates, and the derived numbers an inference engineer cares
// about (GQA ratio, KV-cache bytes/token, decode matmul shapes) — from
// metadata only: safetensors headers (inspect/safetensors_header) or the GGUF
// tensor directory (core/gguf's inspect_gguf). No MLX arrays are created.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/config.h"

namespace mlxforge::inspect {

struct TensorEntry {
  std::string name;                  // canonical key (post sanitize/remap)
  int layer = -1;                    // decoder layer index; -1 = global/vision
  std::string component;             // embed|attn|linear_attn|mlp|moe|norm|lm_head|vision|other
  std::vector<int64_t> shape;        // LOGICAL dims (quantized weights unpacked)
  std::vector<int64_t> stored_shape; // on-disk dims (uint32-packed for MLX quant)
  std::string dtype;                 // "F16", "BF16", "U32", "Q4_K", ...
  std::string quant;                 // "" (dense) | "4b gs64" | "Q4_0" | "type_<id>"
  uint64_t params = 0;               // logical element count (0 for non-param buffers)
  uint64_t bytes = 0;                // on-disk bytes (incl. folded scales/biases)
};

struct ComponentAgg {
  uint64_t params = 0;
  uint64_t bytes = 0;
};

// One decode-step matmul (M=1): activation [1, in] x weight [out, in].
struct MatmulShape {
  std::string name;   // module path within the block ("self_attn.q_proj", ...)
  int64_t in = 0;
  int64_t out = 0;
  std::string quant;  // quant string of the weight ("" = dense)
  std::string note;   // e.g. "8 of 128 experts active"
};

struct ModelSchema {
  // Header card.
  std::string model_name;
  std::string format;         // "safetensors (MLX)" | "GGUF"
  std::string family;         // llama|qwen3|qwen3-moe|qwen3.5-hybrid|qwen3-vl
  std::string quant_summary;  // "4-bit gs64 MLX" | "Q4_K GGUF" | "fp16" | ...
  uint64_t total_params = 0;
  uint64_t total_bytes = 0;   // on-disk weight bytes (incl. dropped buffers)
  uint64_t dropped_bytes = 0; // bytes of buffers the engine never loads
  bool tied_embeddings = false;

  ModelConfig cfg;            // architecture hyperparameters (to_json picks fields)

  // Derived numbers.
  int head_dim = 0;           // resolved (cfg.head_dim, else hidden/n_heads)
  int gqa_ratio = 1;          // n_heads / n_kv_heads
  int n_full_attn_layers = 0; // == n_layers unless hybrid
  double kv_bytes_per_token = 0;  // fp16 cache, full-attention layers only
  std::vector<MatmulShape> decode_matmuls;  // representative block(s), tensor-derived

  std::vector<TensorEntry> tensors;               // sorted by (layer, name)
  std::map<std::string, ComponentAgg> by_component;
  std::vector<ComponentAgg> by_layer;             // size n_layers

  // The JSON blob embedded into the HTML page (shape documented in
  // schematic_html.cpp's template).
  nlohmann::json to_json() const;
};

// Classify a canonical tensor key into its component bucket; exposed for the
// table-driven unit tests.
std::string component_of(const std::string& canonical_key);
// Decoder layer index of a canonical key ("model.layers.N." prefix), -1 if global.
int layer_of(const std::string& canonical_key);

// Build the schema from a safetensors model directory (header parse only).
// `model_name` defaults to the directory basename when empty.
ModelSchema build_schema_from_safetensors(const std::string& model_dir, const ModelConfig& cfg,
                                          const std::string& model_name = "");

// Build the schema from a GGUF file (metadata + tensor directory only).
ModelSchema build_schema_from_gguf(const std::string& gguf_path,
                                   const std::string& model_name = "");

}  // namespace mlxforge::inspect
