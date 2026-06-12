// The schematic HTML renderer: self-contained output (no external refs), the
// embedded schema JSON round-trips, and hostile tensor names cannot break out
// of the data <script> element.
#include <doctest/doctest.h>

#include <string>

#include <nlohmann/json.hpp>

#include "inspect/schematic_html.h"

namespace {

// A minimal-but-complete schema blob (every key the JS template reads).
nlohmann::json sample_schema() {
  return {
      {"header",
       {{"name", "tiny-llama"},
        {"family", "llama"},
        {"format", "safetensors (MLX)"},
        {"quant", "fp16"},
        {"model_type", "llama"},
        {"params", 1000},
        {"bytes", 2000},
        {"dropped_bytes", 0},
        {"context_length", 2048},
        {"vocab", 100},
        {"tied_embeddings", true}}},
      {"arch",
       {{"hidden", 8},
        {"n_layers", 1},
        {"n_heads", 2},
        {"n_kv_heads", 1},
        {"gqa_ratio", 2},
        {"head_dim", 4},
        {"intermediate_size", 16},
        {"qk_norm", false},
        {"moe", nullptr},
        {"hybrid", nullptr},
        {"vision", nullptr},
        {"rope", {{"theta", 10000.0}, {"type", "none"}}}}},
      {"derived",
       {{"kv_bytes_per_token", 32.0},
        {"kv_bytes_per_token_kv8", 17.0},
        {"kv_bytes_per_token_kv4", 9.0},
        {"n_full_attn_layers", 1},
        {"decode_matmuls",
         {{{"name", "self_attn.q_proj"}, {"in", 8}, {"out", 8}, {"quant", ""}, {"note", ""}}}}}},
      {"components", {{"attn", {{"params", 1000}, {"bytes", 2000}}}}},
      {"layers", {{{"idx", 0}, {"kind", "attn"}, {"params", 1000}, {"bytes", 2000}}}},
      {"tensors",
       {{{"name", "model.layers.0.self_attn.q_proj.weight"},
         {"layer", 0},
         {"component", "attn"},
         {"shape", {8, 8}},
         {"dtype", "F16"},
         {"quant", ""},
         {"params", 1000},
         {"bytes", 2000}}}},
  };
}

// The schema JSON sits in <script id="schema-data" type="application/json">.
std::string extract_embedded_json(const std::string& html) {
  const std::string open = "<script id=\"schema-data\" type=\"application/json\">";
  const size_t begin = html.find(open);
  REQUIRE(begin != std::string::npos);
  const size_t start = begin + open.size();
  const size_t end = html.find("</script>", start);
  REQUIRE(end != std::string::npos);
  return html.substr(start, end - start);
}

}  // namespace

TEST_CASE("schematic HTML is a self-contained page with no external references") {
  const std::string html = mlxforge::inspect::render_schematic_html(sample_schema());

  CHECK(html.rfind("<!DOCTYPE html>", 0) == 0);
  CHECK(html.find("</html>") != std::string::npos);
  // Offline-only: no external scripts, styles, fonts or fetches.
  CHECK(html.find("<script src") == std::string::npos);
  CHECK(html.find("<link") == std::string::npos);
  CHECK(html.find("http://") == std::string::npos);
  CHECK(html.find("https://") == std::string::npos);
  CHECK(html.find("@import") == std::string::npos);
  // The splice marker must be gone.
  CHECK(html.find("__MLXFORGE_SCHEMA_JSON__") == std::string::npos);
}

TEST_CASE("embedded schema JSON round-trips") {
  const nlohmann::json schema = sample_schema();
  const std::string html = mlxforge::inspect::render_schematic_html(schema);
  const nlohmann::json parsed = nlohmann::json::parse(extract_embedded_json(html));
  CHECK(parsed == schema);
}

TEST_CASE("hostile tensor names cannot escape the data script element") {
  nlohmann::json schema = sample_schema();
  schema["tensors"][0]["name"] = "</script><script>alert(1)</script>";
  const std::string html = mlxforge::inspect::render_schematic_html(schema);

  // Every '<' inside the blob is escaped, so no early script termination …
  const std::string blob = extract_embedded_json(html);
  CHECK(blob.find('<') == std::string::npos);
  // … and the JSON still round-trips to the original (hostile) name.
  const nlohmann::json parsed = nlohmann::json::parse(blob);
  CHECK(parsed == schema);
}
