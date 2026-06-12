// Metadata-only safetensors header parsing: pure I/O + JSON units over tiny
// synthetic files written into the temp dir. No model or MLX arrays needed.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

#include "inspect/safetensors_header.h"

namespace {

namespace fs = std::filesystem;
using mlxforge::inspect::SafetensorsEntry;

// Write a synthetic .safetensors: 8-byte LE header length + JSON header. The
// data section is zero bytes — the reader never touches it.
void write_safetensors(const std::string& path, const nlohmann::json& header) {
  const std::string h = header.dump();
  const uint64_t len = h.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&len), sizeof(len));
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
}

// A scratch directory that cleans itself up.
struct TempDir {
  fs::path dir;
  explicit TempDir(const std::string& name) : dir(fs::temp_directory_path() / name) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~TempDir() { fs::remove_all(dir); }
  std::string str() const { return dir.string(); }
};

std::map<std::string, SafetensorsEntry> by_name(const std::vector<SafetensorsEntry>& v) {
  std::map<std::string, SafetensorsEntry> m;
  for (const auto& e : v) m.emplace(e.name, e);
  return m;
}

}  // namespace

TEST_CASE("safetensors header parses names, dtypes, shapes and byte sizes") {
  TempDir td("mlxforge_st_header");
  const nlohmann::json header = {
      {"__metadata__", {{"format", "pt"}}},
      {"model.embed_tokens.weight",
       {{"dtype", "BF16"}, {"shape", {128, 64}}, {"data_offsets", {0, 16384}}}},
      {"model.layers.0.self_attn.q_proj.weight",
       {{"dtype", "F16"}, {"shape", {64, 64}}, {"data_offsets", {16384, 24576}}}},
      {"model.norm.weight", {{"dtype", "F32"}, {"shape", {64}}, {"data_offsets", {24576, 24832}}}},
  };
  const std::string file = td.str() + "/model.safetensors";
  write_safetensors(file, header);

  const auto entries = mlxforge::inspect::read_safetensors_header(file);
  CHECK(entries.size() == 3);  // __metadata__ skipped
  const auto m = by_name(entries);

  const auto& embed = m.at("model.embed_tokens.weight");
  CHECK(embed.dtype == "BF16");
  CHECK(embed.shape == std::vector<int64_t>{128, 64});
  CHECK(embed.nbytes == 16384);

  const auto& norm = m.at("model.norm.weight");
  CHECK(norm.dtype == "F32");
  CHECK(norm.shape == std::vector<int64_t>{64});
  CHECK(norm.nbytes == 256);
}

TEST_CASE("safetensors dir merges sharded headers via the index") {
  TempDir td("mlxforge_st_sharded");
  const nlohmann::json index = {{"weight_map",
                                 {{"a.weight", "model-00001-of-00002.safetensors"},
                                  {"b.weight", "model-00002-of-00002.safetensors"}}}};
  std::ofstream(td.str() + "/model.safetensors.index.json") << index.dump();
  write_safetensors(td.str() + "/model-00001-of-00002.safetensors",
                    {{"a.weight", {{"dtype", "F16"}, {"shape", {4}}, {"data_offsets", {0, 8}}}}});
  write_safetensors(td.str() + "/model-00002-of-00002.safetensors",
                    {{"b.weight", {{"dtype", "F16"}, {"shape", {8}}, {"data_offsets", {0, 16}}}}});

  const auto entries = mlxforge::inspect::read_safetensors_dir(td.str());
  CHECK(entries.size() == 2);
  const auto m = by_name(entries);
  CHECK(m.at("a.weight").nbytes == 8);
  CHECK(m.at("b.weight").nbytes == 16);
}

TEST_CASE("safetensors dir falls back to the single file on a stale index") {
  TempDir td("mlxforge_st_stale");
  // The index references a shard that was never downloaded; the consolidated
  // model.safetensors alongside it must win (mirrors load_weights' fallback).
  const nlohmann::json index = {{"weight_map", {{"a.weight", "model-00001-of-00009.safetensors"}}}};
  std::ofstream(td.str() + "/model.safetensors.index.json") << index.dump();
  write_safetensors(td.str() + "/model.safetensors",
                    {{"a.weight", {{"dtype", "F16"}, {"shape", {4}}, {"data_offsets", {0, 8}}}}});

  const auto entries = mlxforge::inspect::read_safetensors_dir(td.str());
  CHECK(entries.size() == 1);
  CHECK(entries[0].name == "a.weight");
}

TEST_CASE("safetensors dir throws when neither layout exists") {
  TempDir td("mlxforge_st_empty");
  CHECK_THROWS_AS(mlxforge::inspect::read_safetensors_dir(td.str()), std::runtime_error);
}

TEST_CASE("safetensors header rejects a corrupt header length") {
  TempDir td("mlxforge_st_corrupt");
  const std::string file = td.str() + "/model.safetensors";
  {
    std::ofstream f(file, std::ios::binary);
    const uint64_t bogus = ~0ull;  // far past the sanity cap
    f.write(reinterpret_cast<const char*>(&bogus), sizeof(bogus));
  }
  CHECK_THROWS_AS(mlxforge::inspect::read_safetensors_header(file), std::runtime_error);
}
