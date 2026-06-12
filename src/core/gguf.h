// Load a self-contained GGUF checkpoint (llama.cpp / Ollama single-file format).
//
// A GGUF file bundles the weights, the model hyperparameters, and the tokenizer
// in one file's metadata, so there is no config.json / tokenizer.json on disk.
// load_gguf_model parses all three via MLX's mx::load_gguf:
//   - weight tensors are remapped from the ggml naming convention
//     ("blk.0.attn_q.weight") to the canonical HF form
//     ("model.layers.0.self_attn.q_proj.weight") and tagged with their quant
//     params (GGUF Q4_0/Q4_1/Q8_0 stay quantized at group_size 32; K-quants and
//     F16/F32 arrive dense fp16);
//   - the ModelConfig is built from the "llama.*" metadata keys (llama-family
//     only — other architectures are rejected);
//   - the tokenizer's raw material (tokens / merges / token types / special ids)
//     is extracted so a BpeTokenizer can be rebuilt without a tokenizer.json.
//
// The llama3 RoPE rescaling is baked into a "rope_freqs.weight" tensor (the
// scaling params are absent from the metadata); its values are lifted into
// ModelConfig::rope_freq_factors and the tensor itself is dropped.
#pragma once

#include <string>
#include <vector>

#include "core/config.h"
#include "core/weights.h"

namespace mlxforge {

// Everything parsed from a GGUF file in a single pass. The tokenizer fields are
// the raw arrays needed to reconstruct a BpeTokenizer (see Tokenizer::from_gguf).
struct GgufModel {
  ModelConfig config;
  Weights weights;
  std::vector<std::string> tokens;       // tokenizer.ggml.tokens (id -> token)
  std::vector<std::string> merges;       // tokenizer.ggml.merges ("L R" per rank)
  std::vector<int> token_types;          // tokenizer.ggml.token_type (per id)
  int bos_id = -1;                        // tokenizer.ggml.bos_token_id
  int eos_id = -1;                        // tokenizer.ggml.eos_token_id
  std::string pre;                        // tokenizer.ggml.pre (e.g. "llama-bpe")
};

// Whether `spec` names a GGUF file (case-insensitive ".gguf" suffix).
bool is_gguf_path(const std::string& spec);

// Parse a GGUF file into config + weights + tokenizer material. Throws on a
// non-llama architecture or malformed/missing required metadata. Loads weight
// tensors (MLX arrays) so must run on the thread that will own the model.
GgufModel load_gguf_model(const std::string& gguf_path);

// Parse only the config + tokenizer material (no weights) from the GGUF
// metadata. Creates no MLX arrays, so it is safe to call on any thread — the
// server uses it on the main thread while the worker loads the weights.
GgufModel load_gguf_config_and_tokenizer(const std::string& gguf_path);

// ----- Metadata-only tensor inspection (CLI `schematic`) --------------------
// The tensor-info section of a GGUF file carries every tensor's name, ggml
// type, dims and data offset; reading it costs a header parse, not a weight
// load. inspect_gguf exposes that directory (plus the config/tokenizer head)
// for model-introspection tooling, creating no MLX arrays.

struct GgufTensorMeta {
  std::string name;            // raw ggml name ("blk.0.attn_q.weight")
  std::string canonical;       // remapped HF key; "" if unrecognized/dropped
  uint32_t ggml_type = 0;
  std::vector<int64_t> shape;  // MLX order (ggml dims reversed); LOGICAL dims
  uint64_t bytes = 0;          // on-disk bytes, derived from the data offsets
};

struct GgufInspection {
  GgufModel head;                      // config + tokenizer material, no weights
  std::vector<GgufTensorMeta> tensors;
  uint64_t file_bytes = 0;
};

// Parse the metadata + tensor directory of a GGUF file. Creates no MLX arrays,
// so it is safe to call on any thread. Throws like load_gguf_config_and_tokenizer
// on an unsupported architecture or a malformed file.
GgufInspection inspect_gguf(const std::string& gguf_path);

// Display name for a ggml tensor type id ("Q4_K", "F16", ...); "type_<id>" for
// an id this build does not know.
std::string ggml_type_name(uint32_t t);

// Effective bits per weight for a ggml tensor type, including the per-block
// scale/min overhead (e.g. Q4_0 = 4.5: 18 bytes per 32 weights); 0 if unknown.
// Display/estimation only — exact byte counts come from the data offsets.
double ggml_bits_per_weight(uint32_t t);

}  // namespace mlxforge
