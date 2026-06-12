#include "inspect/safetensors_header.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/logging.h"
#include "core/weights.h"

namespace mlxforge::inspect {

namespace {
// Headers are typically a few hundred KiB even for 100B+ checkpoints; anything
// past this is a corrupt length field, not a real header.
constexpr uint64_t kMaxHeaderBytes = 256ull * 1024 * 1024;
}  // namespace

std::vector<SafetensorsEntry> read_safetensors_header(const std::string& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) throw std::runtime_error("inspect: cannot open '" + file + "'");

  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
  if (!f || header_len == 0 || header_len > kMaxHeaderBytes) {
    throw std::runtime_error("inspect: corrupt safetensors header length in '" + file + "'");
  }

  std::string header(header_len, '\0');
  f.read(&header[0], static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error("inspect: truncated safetensors header in '" + file + "'");

  nlohmann::json j;
  try {
    j = nlohmann::json::parse(header);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("inspect: bad safetensors header JSON in '" + file + "': " + e.what());
  }

  std::vector<SafetensorsEntry> out;
  out.reserve(j.size());
  for (const auto& [key, v] : j.items()) {
    if (key == "__metadata__") continue;
    SafetensorsEntry e;
    e.name = key;
    e.dtype = v.value("dtype", "");
    if (v.contains("shape")) e.shape = v["shape"].get<std::vector<int64_t>>();
    if (v.contains("data_offsets") && v["data_offsets"].is_array() &&
        v["data_offsets"].size() == 2) {
      const uint64_t begin = v["data_offsets"][0].get<uint64_t>();
      const uint64_t end = v["data_offsets"][1].get<uint64_t>();
      if (end < begin) {
        throw std::runtime_error("inspect: invalid data_offsets for '" + key + "' in '" + file +
                                 "'");
      }
      e.nbytes = end - begin;
    }
    out.push_back(std::move(e));
  }
  return out;
}

std::vector<SafetensorsEntry> read_safetensors_dir(const std::string& model_dir) {
  // Mirror load_weights' shard discovery (core/weights.cpp): prefer the sharded
  // layout, but only when every shard the index names exists — some exports
  // ship a consolidated model.safetensors alongside a stale index.json.
  const std::string index_path = model_dir + "/model.safetensors.index.json";
  std::ifstream index_file(index_path);
  if (index_file) {
    nlohmann::json index_json;
    index_file >> index_json;
    const auto weight_map = parse_shard_index(index_json);
    const auto files = shard_files(weight_map);
    const bool all_present = std::all_of(files.begin(), files.end(), [&](const std::string& f) {
      return std::ifstream(model_dir + "/" + f).good();
    });
    if (all_present) {
      log::debug("inspect: sharded checkpoint, {} files", files.size());
      std::vector<SafetensorsEntry> out;
      for (const auto& file : files) {
        auto shard = read_safetensors_header(model_dir + "/" + file);
        out.insert(out.end(), std::make_move_iterator(shard.begin()),
                   std::make_move_iterator(shard.end()));
      }
      return out;
    }
    log::debug("inspect: index.json shards absent; falling back to single file");
  }
  const std::string single = model_dir + "/model.safetensors";
  if (!std::ifstream(single)) {
    throw std::runtime_error("inspect: no model.safetensors[.index.json] in '" + model_dir + "'");
  }
  return read_safetensors_header(single);
}

}  // namespace mlxforge::inspect
