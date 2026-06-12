// Metadata-only safetensors reader for model inspection.
//
// A .safetensors file starts with an 8-byte little-endian u64 header length
// followed by a JSON header mapping tensor name -> {dtype, shape, data_offsets}.
// Reading just that header recovers every tensor's name/shape/dtype/byte-size
// without touching the weight data — no MLX arrays, no GPU, fast even on the
// multi-GiB shards of a 70B model. Used by the CLI `schematic` command; the
// engine's real weight loading stays in core/weights.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlxforge::inspect {

struct SafetensorsEntry {
  std::string name;            // raw key as stored in the file (not canonicalized)
  std::string dtype;           // header string verbatim ("F16", "BF16", "F32", "U32", ...)
  std::vector<int64_t> shape;  // on-disk dims (packed for MLX-quantized weights)
  uint64_t nbytes = 0;         // data_offsets[1] - data_offsets[0]
};

// Parse one .safetensors file's JSON header. Throws on a missing/unreadable
// file, a corrupt header, or an implausibly large header length.
std::vector<SafetensorsEntry> read_safetensors_header(const std::string& file);

// Read the headers of every shard in a model directory, mirroring
// load_weights' shard discovery: prefer model.safetensors.index.json when all
// the shards it names are present (stale-index exports exist), else fall back
// to the single model.safetensors. Throws if neither layout is found.
std::vector<SafetensorsEntry> read_safetensors_dir(const std::string& model_dir);

}  // namespace mlxforge::inspect
