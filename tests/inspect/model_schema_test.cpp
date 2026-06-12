// ModelSchema builders: quant-triplet folding + packed-shape math, component
// classification, derived KV/GQA math, MoE aggregation, JSON totals. Pure
// units run over synthetic files; the model-gated cases self-skip when the
// cached checkpoints are absent.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "core/config.h"
#include "inspect/model_schema.h"

namespace {

namespace fs = std::filesystem;
using mlxforge::ModelConfig;
using mlxforge::inspect::ModelSchema;
using mlxforge::inspect::TensorEntry;

void write_safetensors(const std::string& path, const nlohmann::json& header) {
  const std::string h = header.dump();
  const uint64_t len = h.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&len), sizeof(len));
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
}

struct TempDir {
  fs::path dir;
  explicit TempDir(const std::string& name) : dir(fs::temp_directory_path() / name) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~TempDir() { fs::remove_all(dir); }
  std::string str() const { return dir.string(); }
};

const TensorEntry& find_tensor(const ModelSchema& s, const std::string& name) {
  for (const auto& e : s.tensors)
    if (e.name == name) return e;
  REQUIRE_MESSAGE(false, "tensor not found: " << name);
  static TensorEntry dummy;
  return dummy;
}

// Append helpers for hand-crafted GGUF files (mirrors tests/core/gguf_test.cpp).
template <typename T>
void put(std::string& b, T v) {
  b.append(reinterpret_cast<const char*>(&v), sizeof(T));
}
void put_str(std::string& b, const std::string& s) {
  put<uint64_t>(b, s.size());
  b += s;
}

}  // namespace

TEST_CASE("component classification buckets canonical keys") {
  using mlxforge::inspect::component_of;
  using mlxforge::inspect::layer_of;
  struct Case {
    const char* key;
    const char* component;
    int layer;
  };
  const Case cases[] = {
      {"model.embed_tokens.weight", "embed", -1},
      {"lm_head.weight", "lm_head", -1},
      {"model.norm.weight", "norm", -1},
      {"model.layers.0.self_attn.q_proj.weight", "attn", 0},
      {"model.layers.3.self_attn.q_norm.weight", "attn", 3},
      {"model.layers.12.input_layernorm.weight", "norm", 12},
      {"model.layers.12.post_attention_layernorm.weight", "norm", 12},
      {"model.layers.5.mlp.gate_proj.weight", "mlp", 5},
      {"model.layers.5.mlp.down_proj.weight", "mlp", 5},
      {"model.layers.7.mlp.gate.weight", "moe", 7},  // router, not gate_proj
      {"model.layers.7.mlp.switch_mlp.up_proj.weight", "moe", 7},
      {"model.layers.7.mlp.experts.31.down_proj.weight", "moe", 7},
      {"model.layers.2.linear_attn.in_proj_qkvz.weight", "linear_attn", 2},
      {"visual.blocks.4.attn.q_proj.weight", "vision", -1},
      {"visual.patch_embed.weight", "vision", -1},
      {"rope_freqs.weight", "other", -1},
  };
  for (const auto& c : cases) {
    CAPTURE(c.key);
    CHECK(component_of(c.key) == c.component);
    CHECK(layer_of(c.key) == c.layer);
  }
}

TEST_CASE("safetensors schema folds quant triplets and unpacks packed shapes") {
  TempDir td("mlxforge_schema_quant");
  // q_proj: 4-bit gs64, logical [256, 512] -> packed weight [256, 64] U32,
  // scales/biases [256, 8] F16. o_proj: an 8-bit gs32 override, logical [4, 32]
  // -> packed [4, 8] U32, scales [4, 1].
  const nlohmann::json header = {
      {"model.embed_tokens.weight",
       {{"dtype", "F16"}, {"shape", {10, 512}}, {"data_offsets", {0, 10240}}}},
      {"model.layers.0.self_attn.q_proj.weight",
       {{"dtype", "U32"}, {"shape", {256, 64}}, {"data_offsets", {10240, 75776}}}},
      {"model.layers.0.self_attn.q_proj.scales",
       {{"dtype", "F16"}, {"shape", {256, 8}}, {"data_offsets", {75776, 79872}}}},
      {"model.layers.0.self_attn.q_proj.biases",
       {{"dtype", "F16"}, {"shape", {256, 8}}, {"data_offsets", {79872, 83968}}}},
      {"model.layers.0.self_attn.o_proj.weight",
       {{"dtype", "U32"}, {"shape", {4, 8}}, {"data_offsets", {83968, 84096}}}},
      {"model.layers.0.self_attn.o_proj.scales",
       {{"dtype", "F16"}, {"shape", {4, 1}}, {"data_offsets", {84096, 84104}}}},
      {"model.norm.weight", {{"dtype", "F16"}, {"shape", {512}}, {"data_offsets", {84104, 85128}}}},
      // A buffer sanitize_key drops; its bytes must still count toward disk size.
      {"model.layers.0.self_attn.rotary_emb.inv_freq",
       {{"dtype", "F32"}, {"shape", {32}}, {"data_offsets", {85128, 85256}}}},
  };
  write_safetensors(td.str() + "/model.safetensors", header);

  ModelConfig cfg;
  cfg.n_layers = 1;
  cfg.hidden = 512;
  cfg.n_heads = 8;
  cfg.n_kv_heads = 4;
  cfg.vocab = 10;
  cfg.quant_group_size = 64;
  cfg.quant_bits = 4;
  cfg.quant_overrides["model.layers.0.self_attn.o_proj"] = {32, 8};

  const ModelSchema s = mlxforge::inspect::build_schema_from_safetensors(td.str(), cfg, "tiny");

  // scales/biases never appear as rows.
  CHECK(s.tensors.size() == 4);

  const auto& q = find_tensor(s, "model.layers.0.self_attn.q_proj.weight");
  CHECK(q.shape == std::vector<int64_t>{256, 512});  // 64 packed cols * 32 / 4 bits
  CHECK(q.stored_shape == std::vector<int64_t>{256, 64});
  CHECK(q.params == 256 * 512);
  CHECK(q.quant == "4b gs64");
  CHECK(q.bytes == 65536 + 4096 + 4096);  // weight + scales + biases
  CHECK(q.component == "attn");

  const auto& o = find_tensor(s, "model.layers.0.self_attn.o_proj.weight");
  CHECK(o.shape == std::vector<int64_t>{4, 32});  // 8 packed cols * 32 / 8 bits
  CHECK(o.quant == "8b gs32");  // override honored

  // No lm_head tensor -> tied embeddings; dropped buffer bytes still counted.
  CHECK(s.tied_embeddings);
  CHECK(s.dropped_bytes == 128);
  CHECK(s.total_bytes == 85256);
  CHECK(s.family == "llama");
  CHECK(s.quant_summary == "4-bit gs64 MLX (mixed)");  // the o_proj 8-bit override

  // Decode matmuls are tensor-derived ([out, in] -> in/out).
  REQUIRE(!s.decode_matmuls.empty());
  CHECK(s.decode_matmuls[0].name == "self_attn.q_proj");
  CHECK(s.decode_matmuls[0].in == 512);
  CHECK(s.decode_matmuls[0].out == 256);
}

TEST_CASE("safetensors schema aggregates raw per-expert MoE tensors") {
  TempDir td("mlxforge_schema_moe");
  const nlohmann::json header = {
      {"model.layers.0.mlp.experts.0.gate_proj.weight",
       {{"dtype", "F16"}, {"shape", {3, 4}}, {"data_offsets", {0, 24}}}},
      {"model.layers.0.mlp.experts.1.gate_proj.weight",
       {{"dtype", "F16"}, {"shape", {3, 4}}, {"data_offsets", {24, 48}}}},
      {"model.layers.0.mlp.gate.weight",
       {{"dtype", "F16"}, {"shape", {2, 4}}, {"data_offsets", {48, 64}}}},
  };
  write_safetensors(td.str() + "/model.safetensors", header);

  ModelConfig cfg;
  cfg.n_layers = 1;
  cfg.hidden = 4;
  cfg.n_heads = 1;
  cfg.n_kv_heads = 1;
  cfg.num_experts = 2;
  cfg.num_experts_per_tok = 2;

  const ModelSchema s = mlxforge::inspect::build_schema_from_safetensors(td.str(), cfg);

  const auto& experts = find_tensor(s, "model.layers.0.mlp.experts.*.gate_proj.weight");
  CHECK(experts.shape == std::vector<int64_t>{2, 3, 4});  // expert count prepended
  CHECK(experts.params == 24);
  CHECK(experts.bytes == 48);
  CHECK(experts.component == "moe");
  CHECK(s.family == "qwen3-moe");
}

TEST_CASE("derived KV math is hybrid-aware") {
  TempDir td("mlxforge_schema_kv");
  write_safetensors(td.str() + "/model.safetensors",
                    {{"model.embed_tokens.weight",
                      {{"dtype", "F16"}, {"shape", {4, 8}}, {"data_offsets", {0, 64}}}}});

  ModelConfig dense;
  dense.n_layers = 16;
  dense.hidden = 2048;
  dense.n_heads = 32;
  dense.n_kv_heads = 8;
  dense.head_dim = 64;
  const ModelSchema sd = mlxforge::inspect::build_schema_from_safetensors(td.str(), dense);
  CHECK(sd.n_full_attn_layers == 16);
  CHECK(sd.gqa_ratio == 4);
  CHECK(sd.kv_bytes_per_token == doctest::Approx(16 * 2 * 8 * 64 * 2));

  ModelConfig hybrid = dense;
  hybrid.n_layers = 48;
  hybrid.full_attention_interval = 4;  // every 4th layer full -> 12 of 48
  const ModelSchema sh = mlxforge::inspect::build_schema_from_safetensors(td.str(), hybrid);
  CHECK(sh.n_full_attn_layers == 12);
  CHECK(sh.kv_bytes_per_token == doctest::Approx(12 * 2 * 8 * 64 * 2));
  CHECK(sh.family == "qwen3.5-hybrid");
}

TEST_CASE("schema JSON totals equal the sum of tensor rows") {
  TempDir td("mlxforge_schema_json");
  const nlohmann::json header = {
      {"model.embed_tokens.weight",
       {{"dtype", "F16"}, {"shape", {10, 8}}, {"data_offsets", {0, 160}}}},
      {"model.layers.0.self_attn.q_proj.weight",
       {{"dtype", "F16"}, {"shape", {8, 8}}, {"data_offsets", {160, 288}}}},
  };
  write_safetensors(td.str() + "/model.safetensors", header);

  ModelConfig cfg;
  cfg.n_layers = 1;
  cfg.hidden = 8;
  cfg.n_heads = 2;
  cfg.n_kv_heads = 2;
  const ModelSchema s = mlxforge::inspect::build_schema_from_safetensors(td.str(), cfg);
  const nlohmann::json j = s.to_json();

  for (const char* key : {"header", "arch", "derived", "components", "layers", "tensors"}) {
    CAPTURE(key);
    CHECK(j.contains(key));
  }

  uint64_t params = 0, bytes = 0;
  for (const auto& row : j["tensors"]) {
    params += row["params"].get<uint64_t>();
    bytes += row["bytes"].get<uint64_t>();
  }
  CHECK(j["header"]["params"].get<uint64_t>() == params);
  CHECK(j["header"]["bytes"].get<uint64_t>() == bytes);  // no dropped buffers here
  CHECK(params == 10 * 8 + 8 * 8);
}

TEST_CASE("GGUF schema reports logical shapes and ggml quant types") {
  // Tiny llama GGUF: one F16 embed + one Q4_0-typed projection.
  constexpr uint32_t kF16 = 1, kQ4_0 = 2;
  constexpr uint64_t kAlign = 32;

  std::string kv;
  uint64_t nkv = 0;
  auto kv_str = [&](const std::string& k, const std::string& v) {
    put_str(kv, k); put<uint32_t>(kv, 8); put_str(kv, v); ++nkv;
  };
  auto kv_u32 = [&](const std::string& k, uint32_t v) {
    put_str(kv, k); put<uint32_t>(kv, 4); put<uint32_t>(kv, v); ++nkv;
  };
  auto kv_f32 = [&](const std::string& k, float v) {
    put_str(kv, k); put<uint32_t>(kv, 6); put<float>(kv, v); ++nkv;
  };
  kv_str("general.architecture", "llama");
  kv_u32("llama.block_count", 1);
  kv_u32("llama.embedding_length", 64);
  kv_u32("llama.attention.head_count", 2);
  kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);

  std::string tinfo;
  put_str(tinfo, "token_embd.weight");
  put<uint32_t>(tinfo, 2);
  put<uint64_t>(tinfo, 64);  // ggml ne: innermost first
  put<uint64_t>(tinfo, 4);
  put<uint32_t>(tinfo, kF16);
  put<uint64_t>(tinfo, 0);
  put_str(tinfo, "blk.0.attn_q.weight");
  put<uint32_t>(tinfo, 2);
  put<uint64_t>(tinfo, 64);
  put<uint64_t>(tinfo, 64);
  put<uint32_t>(tinfo, kQ4_0);
  put<uint64_t>(tinfo, 512);  // 4*64 fp16 = 512 bytes (aligned)

  std::string b;
  put<uint32_t>(b, 0x46554747);
  put<uint32_t>(b, 3);
  put<uint64_t>(b, 2);
  put<uint64_t>(b, nkv);
  b += kv;
  b += tinfo;
  b.append((kAlign - (b.size() % kAlign)) % kAlign, '\0');
  b.append(512, '\0');                    // token_embd: 4*64 fp16
  b.append(64 * 64 / 32 * 18, '\0');      // attn_q: Q4_0, 18 bytes per 32 weights

  const std::string path =
      (fs::temp_directory_path() / "mlxforge_schema_gguf.gguf").string();
  { std::ofstream(path, std::ios::binary).write(b.data(), b.size()); }

  const ModelSchema s = mlxforge::inspect::build_schema_from_gguf(path, "tiny-gguf");
  std::remove(path.c_str());

  CHECK(s.format == "GGUF");
  CHECK(s.model_name == "tiny-gguf");
  CHECK(s.family == "llama");
  CHECK(s.quant_summary == "Q4_0 GGUF");
  CHECK(s.tied_embeddings);  // no output.weight

  const auto& embd = find_tensor(s, "model.embed_tokens.weight");
  CHECK(embd.shape == std::vector<int64_t>{4, 64});  // logical, ne reversed
  CHECK(embd.dtype == "F16");
  CHECK(embd.quant.empty());
  CHECK(embd.params == 256);

  const auto& q = find_tensor(s, "model.layers.0.self_attn.q_proj.weight");
  CHECK(q.shape == std::vector<int64_t>{64, 64});
  CHECK(q.dtype == "Q4_0");
  CHECK(q.quant == "Q4_0");
  CHECK(q.params == 4096);
  CHECK(q.bytes == 64 * 64 / 32 * 18);  // offset-derived: runs to end-of-file
}

TEST_CASE("schematic schema matches the cached 4-bit Llama checkpoint") {
  const std::string dir = MLXFORGE_MODEL_DIR_4BIT;
  if (dir.empty() || !std::ifstream(dir + "/config.json").good()) {
    MESSAGE("MLXFORGE_MODEL_DIR_4BIT not present; skipping");
    return;
  }
  const ModelConfig cfg = ModelConfig::from_file(dir + "/config.json");
  const ModelSchema s = mlxforge::inspect::build_schema_from_safetensors(dir, cfg);

  CHECK(s.family == "llama");
  CHECK(s.quant_summary.find("4-bit") != std::string::npos);
  CHECK(s.tied_embeddings);
  // Llama-3.2-1B is ~1.24B parameters; the schema's logical count must land
  // within 2% (the packed-shape unpacking is what this gates).
  CHECK(s.total_params > 1.21e9);
  CHECK(s.total_params < 1.26e9);
  // Every per-layer tensor classifies into a real bucket.
  for (const auto& e : s.tensors) {
    CAPTURE(e.name);
    if (e.layer >= 0) CHECK(e.component != "other");
  }
}

TEST_CASE("schematic schema reads the cached GGUF checkpoint") {
  const std::string path = MLXFORGE_GGUF_MODEL;
  if (path.empty() || !std::ifstream(path).good()) {
    MESSAGE("MLXFORGE_GGUF_MODEL not present; skipping");
    return;
  }
  const ModelSchema s = mlxforge::inspect::build_schema_from_gguf(path);
  CHECK(s.family == "llama");
  CHECK(s.cfg.n_layers == 16);
  CHECK(s.by_layer.size() == 16);
  CHECK(s.total_params > 1.21e9);
  CHECK(s.total_params < 1.26e9);
  for (const auto& e : s.tensors) {
    CAPTURE(e.name);
    if (e.layer >= 0) CHECK(e.component != "other");
  }
}
