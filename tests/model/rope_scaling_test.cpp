// RoPE-scaling unit gates (no model download needed): the yarn/linear frequency
// schedules and the yarn mscale vs the mlx-lm fixtures (fixture-to-fixture — the
// committed q_pre0 tensor pushed through our freqs must reproduce the committed
// q_rope0), plus the validation/override behavior: unknown rope_type values and
// unsupported model shapes must throw, never silently fall back to unscaled RoPE.
#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <optional>
#include <vector>

#include "core/config.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "model/decoder_model.h"
#include "support/reference.h"

using namespace mlxforge::test;

namespace {

bool yarn_fixtures_present() {
  return std::ifstream(qwen3_yarn_ref_path("manifest.json")).good();
}

// Minimal config in the Qwen3-0.6B rope shape (the only fields
// compute_rope_setup reads besides rope_scaling).
mlxforge::ModelConfig base_cfg() {
  mlxforge::ModelConfig c;
  c.head_dim = 128;
  c.rope_theta = 1000000.0f;
  c.max_position_embeddings = 40960;
  return c;
}

// The injected yarn recipe from dump_ref.py's qwen3_yarn spec.
mlxforge::RopeScaling yarn_scaling() {
  mlxforge::RopeScaling rs;
  rs.rope_type = "yarn";
  rs.factor = 4.0f;
  rs.original_max_position_embeddings = 32768;
  return rs;
}

// Plain base**(2i/d) schedule, the unscaled reference for the no-op cases.
mx::array plain_freqs(const mlxforge::ModelConfig& c) {
  mx::array idx = mx::arange(0, c.head_dim, 2, mx::float32);
  mx::array freqs = mx::power(mx::array(c.rope_theta),
                              mx::divide(idx, mx::array(static_cast<float>(c.head_dim))));
  mx::eval(freqs);
  return freqs;
}

mx::array rope_with_freqs(const mx::array& x, const mx::array& freqs) {
  return mx::fast::rope(x, x.shape().back(), /*traditional=*/false, /*base=*/std::nullopt,
                        /*scale=*/1.0f, /*offset=*/0, freqs);
}

}  // namespace

TEST_CASE("yarn freqs + mscale match mlx-lm's YarnRoPE") {
  if (!yarn_fixtures_present()) {
    MESSAGE("fixtures_qwen3_yarn not present; skipping");
    return;
  }
  mlxforge::ModelConfig c = base_cfg();
  c.rope_scaling = yarn_scaling();
  mlxforge::RopeSetup setup = mlxforge::compute_rope_setup(c);

  // Both sides compute the schedule in float32 from the same formula, so the
  // tolerance is far tighter than the fp16 tensor gates.
  assert_close(setup.freqs, load_qwen3_yarn_npy("rope_freqs.npy"), /*rtol=*/1e-5f,
               /*atol=*/1e-3f);

  mx::array ref_mscale = load_qwen3_yarn_npy("rope_mscale.npy");
  mx::eval(ref_mscale);
  CHECK(setup.mscale == doctest::Approx(ref_mscale.item<float>()).epsilon(1e-6));
  // Qwen3 yarn defaults (mscale=1, mscale_all_dim=0) reduce to 0.1*ln(factor)+1.
  CHECK(setup.mscale == doctest::Approx(0.1f * std::log(4.0f) + 1.0f).epsilon(1e-6));
}

TEST_CASE("yarn rope application reproduces the reference q_rope0/k_rope0") {
  if (!yarn_fixtures_present()) {
    MESSAGE("fixtures_qwen3_yarn not present; skipping");
    return;
  }
  mlxforge::ModelConfig c = base_cfg();
  c.rope_scaling = yarn_scaling();
  mlxforge::RopeSetup setup = mlxforge::compute_rope_setup(c);

  // Fixture-to-fixture: mscale * q_pre0 through fast::rope with our freqs must
  // equal mlx-lm's attn.rope(q) (YarnRoPE scales its input by mscale).
  mx::array q = load_qwen3_yarn_npy("q_pre0.npy");
  mx::array qs = mx::multiply(q, mx::array(setup.mscale, q.dtype()));
  assert_close(rope_with_freqs(qs, setup.freqs), load_qwen3_yarn_npy("q_rope0.npy"));
}

TEST_CASE("linear rope matches mlx-lm's nn.RoPE(scale=1/factor)") {
  if (!yarn_fixtures_present()) {
    MESSAGE("fixtures_qwen3_yarn not present; skipping");
    return;
  }
  mlxforge::ModelConfig c = base_cfg();
  mlxforge::RopeScaling rs;
  rs.rope_type = "linear";
  rs.factor = 4.0f;  // the factor dump_ref.py's linear oracle uses
  c.rope_scaling = rs;
  mlxforge::RopeSetup setup = mlxforge::compute_rope_setup(c);
  CHECK(setup.mscale == 1.0f);

  mx::array q = load_qwen3_yarn_npy("q_pre0.npy");
  assert_close(rope_with_freqs(q, setup.freqs), load_qwen3_yarn_npy("q_rope0_linear.npy"));
}

TEST_CASE("default / missing rope_scaling keep the plain schedule") {
  mlxforge::ModelConfig c = base_cfg();
  mlxforge::RopeSetup none = mlxforge::compute_rope_setup(c);
  CHECK(none.mscale == 1.0f);
  assert_close(none.freqs, plain_freqs(c), /*rtol=*/1e-6f, /*atol=*/1e-3f);

  mlxforge::RopeScaling rs;
  rs.rope_type = "default";
  c.rope_scaling = rs;
  mlxforge::RopeSetup dflt = mlxforge::compute_rope_setup(c);
  CHECK(dflt.mscale == 1.0f);
  assert_close(dflt.freqs, plain_freqs(c), /*rtol=*/1e-6f, /*atol=*/1e-3f);
}

TEST_CASE("unknown rope_type values are rejected, never silently unscaled") {
  mlxforge::ModelConfig c = base_cfg();
  mlxforge::RopeScaling rs;
  rs.factor = 2.0f;
  rs.original_max_position_embeddings = 4096;

  for (const char* t : {"dynamic", "longrope", "su"}) {
    rs.rope_type = t;
    c.rope_scaling = rs;
    CHECK_THROWS_WITH_AS(mlxforge::validate_rope_scaling(c),
                         ("unsupported rope_scaling rope_type '" + std::string(t) +
                          "' (supported: default, llama3, yarn, linear)")
                             .c_str(),
                         std::runtime_error);
    // compute_rope_setup re-validates (defense for direct construction paths).
    CHECK_THROWS_AS(mlxforge::compute_rope_setup(c), std::runtime_error);
  }
}

TEST_CASE("yarn/linear are rejected off the shared full-rotary text path") {
  mlxforge::RopeScaling rs = yarn_scaling();

  // Hybrid (Qwen3.5): own partial rope, never reads the shared freqs.
  mlxforge::ModelConfig hybrid = base_cfg();
  hybrid.full_attention_interval = 4;
  hybrid.rope_scaling = rs;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(hybrid), std::runtime_error);

  mlxforge::ModelConfig partial = base_cfg();
  partial.partial_rotary_factor = 0.5f;
  partial.rope_scaling = rs;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(partial), std::runtime_error);

  // Vision / M-RoPE (Qwen3-VL): 3D positions, hand-rolled rotation.
  mlxforge::ModelConfig vision = base_cfg();
  vision.vision = mlxforge::VisionConfig{};
  vision.rope_scaling = rs;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(vision), std::runtime_error);

  mlxforge::ModelConfig mrope = base_cfg();
  mrope.mrope_section = {24, 20, 20};
  mrope.rope_scaling = rs;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(mrope), std::runtime_error);

  // GGUF checkpoints bake llama3 factors into rope_freqs.weight — conflicting.
  mlxforge::ModelConfig gguf = base_cfg();
  gguf.rope_freq_factors = std::vector<float>(64, 1.0f);
  gguf.rope_scaling = rs;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(gguf), std::runtime_error);

  // Required parameters.
  mlxforge::ModelConfig no_orig = base_cfg();
  mlxforge::RopeScaling bad = rs;
  bad.original_max_position_embeddings = 0;
  no_orig.rope_scaling = bad;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(no_orig), std::runtime_error);

  mlxforge::ModelConfig bad_factor = base_cfg();
  bad = rs;
  bad.factor = 0.0f;
  bad_factor.rope_scaling = bad;
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(bad_factor), std::runtime_error);
}

TEST_CASE("apply_rope_scaling_override: replacement semantics and defaults") {
  mlxforge::ModelConfig c = base_cfg();

  // Malformed JSON / non-object values are clear errors.
  CHECK_THROWS_AS(mlxforge::apply_rope_scaling_override(c, "{not json"), std::runtime_error);
  CHECK_THROWS_AS(mlxforge::apply_rope_scaling_override(c, "42"), std::runtime_error);

  // A yarn override without the original window scales the checkpoint's
  // shipped context (max_position_embeddings).
  mlxforge::apply_rope_scaling_override(c, R"({"rope_type":"yarn","factor":4.0})");
  REQUIRE(c.rope_scaling.has_value());
  CHECK(c.rope_scaling->rope_type == "yarn");
  CHECK(c.rope_scaling->factor == 4.0f);
  CHECK(c.rope_scaling->original_max_position_embeddings == 40960);
  CHECK(c.rope_scaling->beta_fast == 32.0f);  // mlx-lm YarnRoPE defaults
  CHECK(c.rope_scaling->beta_slow == 1.0f);
  mlxforge::validate_rope_scaling(c);  // the resulting config is valid

  // Full replacement (vLLM semantics), and the legacy "type" key is accepted.
  mlxforge::apply_rope_scaling_override(c, R"({"type":"linear","factor":2.0})");
  CHECK(c.rope_scaling->rope_type == "linear");
  CHECK(c.rope_scaling->factor == 2.0f);

  // An unknown type in an override survives parsing but fails validation —
  // the engine calls validate right after.
  mlxforge::apply_rope_scaling_override(c, R"({"rope_type":"dynamic","factor":2.0})");
  CHECK_THROWS_AS(mlxforge::validate_rope_scaling(c), std::runtime_error);
}
