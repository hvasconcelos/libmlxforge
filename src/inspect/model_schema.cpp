#include "inspect/model_schema.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <map>
#include <set>

#include "core/gguf.h"
#include "core/logging.h"
#include "core/weights.h"
#include "inspect/safetensors_header.h"

namespace mlxforge::inspect {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
bool starts_with(const std::string& s, const std::string& prefix) {
  return s.rfind(prefix, 0) == 0;
}
bool contains(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

std::string path_basename(const std::string& path) {
  std::string p = path;
  while (!p.empty() && p.back() == '/') p.pop_back();
  const size_t slash = p.find_last_of('/');
  return slash == std::string::npos ? p : p.substr(slash + 1);
}

uint64_t prod(const std::vector<int64_t>& shape) {
  uint64_t n = 1;
  for (int64_t d : shape) n *= static_cast<uint64_t>(d);
  return n;
}

// Preferred in-block display order for the decode-matmul list (q before k
// before v..., not alphabetical). Unknown modules sort after, by name.
int module_rank(const std::string& module) {
  static const std::pair<const char*, int> kRanks[] = {
      {"self_attn.q_proj", 0}, {"self_attn.k_proj", 1}, {"self_attn.v_proj", 2},
      {"self_attn.o_proj", 3}, {"mlp.gate.", 4},        {"mlp.gate_proj", 5},
      {"mlp.up_proj", 6},      {"mlp.down_proj", 7},
      {"mlp.switch_mlp.gate_proj", 5}, {"mlp.switch_mlp.up_proj", 6},
      {"mlp.switch_mlp.down_proj", 7},
  };
  for (const auto& [key, rank] : kRanks)
    if (contains(module, key)) return rank;
  return 8;
}

}  // namespace

std::string component_of(const std::string& key) {
  if (starts_with(key, "visual.")) return "vision";
  if (starts_with(key, "model.embed_tokens")) return "embed";
  if (starts_with(key, "lm_head")) return "lm_head";
  if (starts_with(key, "model.norm")) return "norm";
  if (starts_with(key, "model.layers.")) {
    if (contains(key, ".self_attn.")) return "attn";
    if (contains(key, ".linear_attn.")) return "linear_attn";
    // MoE pieces: the stacked experts (switch_mlp), raw per-expert tensors,
    // the router (".mlp.gate." — distinct from the dense ".mlp.gate_proj."),
    // and any shared expert.
    if (contains(key, ".mlp.switch_mlp.") || contains(key, ".mlp.experts.") ||
        contains(key, ".mlp.gate.") || contains(key, ".mlp.shared_expert")) {
      return "moe";
    }
    if (contains(key, "layernorm") || contains(key, ".norm")) return "norm";
    if (contains(key, ".mlp.")) return "mlp";
    return "other";
  }
  return "other";
}

int layer_of(const std::string& key) {
  static const std::string kPrefix = "model.layers.";
  if (!starts_with(key, kPrefix)) return -1;
  const size_t dot = key.find('.', kPrefix.size());
  if (dot == std::string::npos) return -1;
  const std::string idx = key.substr(kPrefix.size(), dot - kPrefix.size());
  if (idx.empty() || !std::all_of(idx.begin(), idx.end(),
                                  [](unsigned char c) { return std::isdigit(c); })) {
    return -1;
  }
  return std::stoi(idx);
}

namespace {

// Collapse raw per-expert MoE tensors ("...mlp.experts.<e>.gate_proj.weight")
// into one row per (layer, proj): "...mlp.experts.*.gate_proj.weight" with the
// expert count prepended to the shape. Keeps the table readable on 128-expert
// models; pre-stacked switch_mlp checkpoints pass through untouched.
std::vector<TensorEntry> aggregate_per_expert(std::vector<TensorEntry> in) {
  struct Agg {
    TensorEntry entry;  // first expert's entry, name rewritten
    uint64_t count = 0;
  };
  std::map<std::string, Agg> aggs;
  std::vector<TensorEntry> out;
  out.reserve(in.size());

  for (auto& e : in) {
    const size_t exp_pos = e.name.find(".experts.");
    if (exp_pos == std::string::npos) {
      out.push_back(std::move(e));
      continue;
    }
    const size_t idx_begin = exp_pos + std::strlen(".experts.");
    size_t idx_end = idx_begin;
    while (idx_end < e.name.size() && std::isdigit(static_cast<unsigned char>(e.name[idx_end])))
      ++idx_end;
    if (idx_end == idx_begin || idx_end >= e.name.size() || e.name[idx_end] != '.') {
      out.push_back(std::move(e));  // not the per-expert pattern
      continue;
    }
    const std::string agg_name =
        e.name.substr(0, idx_begin) + "*" + e.name.substr(idx_end);
    auto [it, fresh] = aggs.try_emplace(agg_name);
    if (fresh) {
      it->second.entry = e;
      it->second.entry.name = agg_name;
    } else {
      it->second.entry.params += e.params;
      it->second.entry.bytes += e.bytes;
    }
    ++it->second.count;
  }

  for (auto& [_, agg] : aggs) {
    agg.entry.shape.insert(agg.entry.shape.begin(), static_cast<int64_t>(agg.count));
    if (!agg.entry.stored_shape.empty())
      agg.entry.stored_shape.insert(agg.entry.stored_shape.begin(),
                                    static_cast<int64_t>(agg.count));
    out.push_back(std::move(agg.entry));
  }
  return out;
}

// Family label, mirroring create_model's dispatch (model/model_factory.cpp).
std::string detect_family(const ModelConfig& cfg, bool has_qk_norm) {
  if (cfg.full_attention_interval > 0) return "qwen3.5-hybrid";
  if (cfg.num_experts > 0) return "qwen3-moe";
  if (cfg.has_vision_tower()) return "qwen3-vl";
  if (has_qk_norm) return "qwen3";
  return "llama";
}

// Aggregates, totals, sort order, derived math — shared by both builders.
void finalize(ModelSchema& s) {
  const ModelConfig& cfg = s.cfg;
  s.head_dim = cfg.head_dim > 0 ? cfg.head_dim : (cfg.n_heads > 0 ? cfg.hidden / cfg.n_heads : 0);
  s.gqa_ratio = cfg.n_kv_heads > 0 ? cfg.n_heads / cfg.n_kv_heads : 1;
  s.n_full_attn_layers = 0;
  for (int i = 0; i < cfg.n_layers; ++i)
    if (!cfg.is_linear_layer(i)) ++s.n_full_attn_layers;
  // fp16 cache: K + V per full-attention layer, 2 bytes per element.
  s.kv_bytes_per_token =
      static_cast<double>(s.n_full_attn_layers) * 2.0 * cfg.n_kv_heads * s.head_dim * 2.0;

  bool has_qk_norm = false;
  for (const auto& e : s.tensors)
    if (contains(e.name, ".self_attn.q_norm.")) { has_qk_norm = true; break; }
  s.family = detect_family(cfg, has_qk_norm);
  s.tied_embeddings = std::none_of(s.tensors.begin(), s.tensors.end(), [](const TensorEntry& e) {
    return starts_with(e.name, "lm_head.");
  });

  std::sort(s.tensors.begin(), s.tensors.end(), [](const TensorEntry& a, const TensorEntry& b) {
    if (a.layer != b.layer) return a.layer < b.layer;
    return a.name < b.name;
  });

  s.total_params = 0;
  s.total_bytes = s.dropped_bytes;
  s.by_component.clear();
  s.by_layer.assign(std::max(0, cfg.n_layers), ComponentAgg{});
  for (const auto& e : s.tensors) {
    s.total_params += e.params;
    s.total_bytes += e.bytes;
    auto& comp = s.by_component[e.component];
    comp.params += e.params;
    comp.bytes += e.bytes;
    if (e.layer >= 0 && e.layer < static_cast<int>(s.by_layer.size())) {
      s.by_layer[e.layer].params += e.params;
      s.by_layer[e.layer].bytes += e.bytes;
    }
  }

  // Decode matmul shapes (M=1) from the actual tensors of representative
  // layers — the first full-attention layer, plus the first linear-attention
  // layer for hybrid models. Tensor-derived dims catch config/checkpoint drift
  // (and the attn_output_gate 2x q_proj) for free.
  s.decode_matmuls.clear();
  std::vector<int> reps;
  for (int i = 0; i < cfg.n_layers; ++i)
    if (!cfg.is_linear_layer(i)) { reps.push_back(i); break; }
  for (int i = 0; i < cfg.n_layers; ++i)
    if (cfg.is_linear_layer(i)) { reps.push_back(i); break; }
  for (int rep : reps) {
    const std::string prefix = "model.layers." + std::to_string(rep) + ".";
    std::vector<MatmulShape> block;
    for (const auto& e : s.tensors) {
      if (e.layer != rep || e.shape.size() < 2) continue;
      if (e.component == "norm" || e.component == "other") continue;
      MatmulShape m;
      m.name = e.name.substr(prefix.size());
      if (ends_with(m.name, ".weight")) m.name.resize(m.name.size() - std::strlen(".weight"));
      m.in = e.shape.back();
      m.out = e.shape[e.shape.size() - 2];
      m.quant = e.quant;
      if (e.shape.size() == 3) {  // stacked experts (E, out, in)
        m.note = std::to_string(cfg.num_experts_per_tok) + " of " +
                 std::to_string(e.shape.front()) + " experts active";
      } else if (contains(m.name, "mlp.gate") && !contains(m.name, "gate_proj")) {
        m.note = "router";
      } else if (cfg.is_linear_layer(rep)) {
        m.note = "linear attention";
      }
      block.push_back(std::move(m));
    }
    std::stable_sort(block.begin(), block.end(), [](const MatmulShape& a, const MatmulShape& b) {
      return module_rank(a.name) < module_rank(b.name);
    });
    s.decode_matmuls.insert(s.decode_matmuls.end(), block.begin(), block.end());
  }
}

// Majority vote over the labels of the big (2-D+) layer weights, so norms and
// odd buffers never skew the summary; `mixed` reports a heterogeneous file.
std::string majority_label(const std::vector<std::string>& labels, bool& mixed) {
  mixed = false;
  if (labels.empty()) return "";
  std::map<std::string, int> counts;
  for (const auto& l : labels) ++counts[l];
  std::string best;
  int best_n = -1;
  for (const auto& [label, n] : counts)
    if (n > best_n) { best = label; best_n = n; }
  mixed = counts.size() > 1;
  return best;
}

}  // namespace

ModelSchema build_schema_from_safetensors(const std::string& model_dir, const ModelConfig& cfg,
                                          const std::string& model_name) {
  ModelSchema s;
  s.cfg = cfg;
  s.format = "safetensors (MLX)";
  s.model_name = model_name.empty() ? path_basename(model_dir) : model_name;

  // Canonicalize keys the same way the loader does, keeping the vision tower
  // unconditionally — the schematic should show it even when the engine's
  // text-only load would drop it. Dropped buffers still count toward bytes.
  std::map<std::string, SafetensorsEntry> canon;
  for (auto& e : read_safetensors_dir(model_dir)) {
    auto key = sanitize_key(e.name, /*keep_vision=*/true);
    if (!key) {
      s.dropped_bytes += e.nbytes;
      continue;
    }
    e.name = *key;
    canon.emplace(*key, std::move(e));
  }

  // A "<base>.scales" sibling marks "<base>.weight" as MLX-quantized; the
  // triplet folds into one logical tensor with the packed shape unpacked.
  std::set<std::string> quant_bases;
  for (const auto& [name, _] : canon) {
    if (!ends_with(name, ".scales")) continue;
    const std::string base = name.substr(0, name.size() - std::strlen(".scales"));
    if (canon.count(base + ".weight")) quant_bases.insert(base);
  }

  // A "<base>.scales"/".biases" of a quantized base folds into the base's
  // ".weight" row below, so it never becomes a row of its own.
  auto is_folded_quant_sibling = [&](const std::string& name) {
    for (const char* suf : {".scales", ".biases"}) {
      if (ends_with(name, suf) &&
          quant_bases.count(name.substr(0, name.size() - std::strlen(suf)))) {
        return true;
      }
    }
    return false;
  };

  std::vector<std::string> quant_labels, dense_labels;
  for (const auto& [name, e] : canon) {
    if (is_folded_quant_sibling(name)) continue;

    TensorEntry t;
    t.name = name;
    t.layer = layer_of(name);
    t.component = component_of(name);
    t.dtype = e.dtype;
    t.shape = e.shape;
    t.bytes = e.nbytes;

    const bool is_weight = ends_with(name, ".weight");
    const std::string base =
        is_weight ? name.substr(0, name.size() - std::strlen(".weight")) : "";
    if (is_weight && quant_bases.count(base)) {
      const QuantParams qp = cfg.quant_for(base);
      t.stored_shape = e.shape;
      // Packed uint32 columns: in * bits / 32 -> logical in = cols * 32 / bits.
      t.shape.back() = e.shape.back() * 32 / qp.bits;
      const auto& scales = canon.at(base + ".scales");
      const int64_t scales_in = scales.shape.empty() ? 0 : scales.shape.back();
      if (scales_in * qp.group_size != t.shape.back()) {
        log::warn(
            "inspect: quant shape mismatch for '{}': packed-derived in={} vs "
            "scales-derived in={} (bits={} gs={}) — check quant config",
            base, t.shape.back(), scales_in * qp.group_size, qp.bits, qp.group_size);
      }
      t.quant = std::to_string(qp.bits) + "b gs" + std::to_string(qp.group_size);
      t.bytes += scales.nbytes;
      if (auto it = canon.find(base + ".biases"); it != canon.end())
        t.bytes += it->second.nbytes;
      if (t.shape.size() >= 2)
        quant_labels.push_back(std::to_string(qp.bits) + "-bit gs" +
                               std::to_string(qp.group_size));
    } else if (t.shape.size() >= 2 && t.layer >= 0) {
      std::string d = e.dtype;
      std::transform(d.begin(), d.end(), d.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      dense_labels.push_back(d == "f16" ? "fp16" : d == "bf16" ? "bf16" : d);
    }
    t.params = prod(t.shape);
    s.tensors.push_back(std::move(t));
  }

  s.tensors = aggregate_per_expert(std::move(s.tensors));
  bool mixed = false;
  s.quant_summary = !quant_labels.empty() ? majority_label(quant_labels, mixed) + " MLX"
                                          : majority_label(dense_labels, mixed);
  if (s.quant_summary.empty()) s.quant_summary = "fp16";
  if (mixed) s.quant_summary += " (mixed)";
  finalize(s);
  return s;
}

ModelSchema build_schema_from_gguf(const std::string& gguf_path, const std::string& model_name) {
  GgufInspection insp = inspect_gguf(gguf_path);

  ModelSchema s;
  s.cfg = insp.head.config;
  s.format = "GGUF";
  s.model_name = model_name.empty() ? path_basename(gguf_path) : model_name;

  std::vector<std::string> quant_labels;
  for (const auto& tm : insp.tensors) {
    TensorEntry t;
    t.name = tm.canonical.empty() ? tm.name : tm.canonical;
    t.layer = layer_of(t.name);
    t.component = component_of(t.name);
    t.shape = tm.shape;  // GGUF dims are already logical
    t.dtype = ggml_type_name(tm.ggml_type);
    t.bytes = tm.bytes;
    const bool dense = tm.ggml_type == 0 /*F32*/ || tm.ggml_type == 1 /*F16*/ ||
                       tm.ggml_type == 30 /*BF16*/;
    if (!dense) t.quant = t.dtype;
    // rope_freqs is a baked frequency table, not model parameters.
    t.params = (tm.name == "rope_freqs.weight") ? 0 : prod(t.shape);
    if (t.shape.size() >= 2 && t.layer >= 0) quant_labels.push_back(t.dtype);
    s.tensors.push_back(std::move(t));
  }

  bool mixed = false;
  s.quant_summary = majority_label(quant_labels, mixed);
  if (s.quant_summary.empty()) s.quant_summary = "F16";
  s.quant_summary += " GGUF";
  if (mixed) s.quant_summary += " (mixed)";
  s.cfg.tie_word_embeddings = std::none_of(
      s.tensors.begin(), s.tensors.end(),
      [](const TensorEntry& e) { return starts_with(e.name, "lm_head."); });
  finalize(s);
  return s;
}

nlohmann::json ModelSchema::to_json() const {
  nlohmann::json j;
  j["header"] = {
      {"name", model_name},
      {"family", family},
      {"format", format},
      {"quant", quant_summary},
      {"model_type", cfg.model_type},
      {"params", total_params},
      {"bytes", total_bytes},
      {"dropped_bytes", dropped_bytes},
      {"context_length", cfg.max_position_embeddings},
      {"vocab", cfg.vocab},
      {"tied_embeddings", tied_embeddings},
  };

  nlohmann::json arch = {
      {"hidden", cfg.hidden},
      {"n_layers", cfg.n_layers},
      {"n_heads", cfg.n_heads},
      {"n_kv_heads", cfg.n_kv_heads},
      {"gqa_ratio", gqa_ratio},
      {"head_dim", head_dim},
      {"intermediate_size", cfg.intermediate_size},
  };
  bool has_qk_norm = false;
  for (const auto& e : tensors)
    if (e.name.find(".self_attn.q_norm.") != std::string::npos) { has_qk_norm = true; break; }
  arch["qk_norm"] = has_qk_norm;
  if (cfg.num_experts > 0) {
    int n_moe = 0;
    for (int i = 0; i < cfg.n_layers; ++i)
      if (cfg.is_moe_layer(i)) ++n_moe;
    arch["moe"] = {{"experts", cfg.num_experts},
                   {"top_k", cfg.num_experts_per_tok},
                   {"moe_intermediate", cfg.moe_intermediate_size},
                   {"n_moe_layers", n_moe}};
  } else {
    arch["moe"] = nullptr;
  }
  if (cfg.full_attention_interval > 0) {
    arch["hybrid"] = {{"full_attention_interval", cfg.full_attention_interval},
                      {"n_linear_layers", cfg.n_layers - n_full_attn_layers},
                      {"linear_num_key_heads", cfg.linear_num_key_heads},
                      {"linear_num_value_heads", cfg.linear_num_value_heads},
                      {"linear_key_head_dim", cfg.linear_key_head_dim},
                      {"linear_value_head_dim", cfg.linear_value_head_dim},
                      {"conv_kernel", cfg.linear_conv_kernel_dim}};
  } else {
    arch["hybrid"] = nullptr;
  }
  {
    nlohmann::json rope = {{"theta", cfg.rope_theta}};
    if (cfg.rope_scaling.has_value()) {
      rope["type"] = cfg.rope_scaling->rope_type;
      rope["factor"] = cfg.rope_scaling->factor;
    } else if (cfg.rope_freq_factors.has_value()) {
      rope["type"] = "llama3 (baked)";
    } else {
      rope["type"] = "none";
    }
    arch["rope"] = std::move(rope);
  }
  if (cfg.vision.has_value()) {
    const VisionConfig& v = *cfg.vision;
    arch["vision"] = {{"depth", v.depth},
                      {"hidden", v.hidden},
                      {"intermediate_size", v.intermediate_size},
                      {"num_heads", v.num_heads},
                      {"patch_size", v.patch_size},
                      {"spatial_merge_size", v.spatial_merge_size},
                      {"out_hidden_size", v.out_hidden_size},
                      {"deepstack_indexes", v.deepstack_visual_indexes}};
  } else {
    arch["vision"] = nullptr;
  }
  j["arch"] = std::move(arch);

  // Quantized-KV variants mirror cache/kv_quant's layout: packed values plus
  // fp16 scale+bias per group of 64 — approximate (group padding ignored).
  auto kv_quant_bytes = [&](int bits) {
    return static_cast<double>(n_full_attn_layers) * 2.0 * cfg.n_kv_heads * head_dim *
           (bits / 8.0 + 4.0 / 64.0);
  };
  nlohmann::json matmuls = nlohmann::json::array();
  for (const auto& m : decode_matmuls) {
    matmuls.push_back({{"name", m.name},
                       {"in", m.in},
                       {"out", m.out},
                       {"quant", m.quant},
                       {"note", m.note}});
  }
  j["derived"] = {{"kv_bytes_per_token", kv_bytes_per_token},
                  {"kv_bytes_per_token_kv8", kv_quant_bytes(8)},
                  {"kv_bytes_per_token_kv4", kv_quant_bytes(4)},
                  {"n_full_attn_layers", n_full_attn_layers},
                  {"decode_matmuls", std::move(matmuls)}};

  nlohmann::json components = nlohmann::json::object();
  for (const auto& [name, agg] : by_component)
    components[name] = {{"params", agg.params}, {"bytes", agg.bytes}};
  j["components"] = std::move(components);

  nlohmann::json layers = nlohmann::json::array();
  for (size_t i = 0; i < by_layer.size(); ++i) {
    const int idx = static_cast<int>(i);
    const char* kind = cfg.is_linear_layer(idx) ? "linear"
                       : cfg.is_moe_layer(idx)  ? "moe"
                                                : "attn";
    layers.push_back({{"idx", idx},
                      {"kind", kind},
                      {"params", by_layer[i].params},
                      {"bytes", by_layer[i].bytes}});
  }
  j["layers"] = std::move(layers);

  nlohmann::json rows = nlohmann::json::array();
  for (const auto& e : tensors) {
    nlohmann::json row = {{"name", e.name},     {"layer", e.layer}, {"component", e.component},
                          {"shape", e.shape},   {"dtype", e.dtype}, {"quant", e.quant},
                          {"params", e.params}, {"bytes", e.bytes}};
    if (!e.stored_shape.empty() && e.stored_shape != e.shape)
      row["stored_shape"] = e.stored_shape;
    rows.push_back(std::move(row));
  }
  j["tensors"] = std::move(rows);
  return j;
}

}  // namespace mlxforge::inspect
