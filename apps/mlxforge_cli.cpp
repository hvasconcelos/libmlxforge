// mlxforge-cli — command-line entry point.
//
// Usage patterns:
//   mlxforge-cli
//     - Performs a smoke test: adds two arrays, evaluates the computation, and prints the sum.
//   mlxforge-cli dump-weights <dir>
//     - Loads a model's weights from the supplied directory, prints key/shape/dtype for each tensor,
//       asserts that all tensors are fp16, and reports the peak resident memory used.
//   mlxforge-cli generate <model> <prompt> [max_tokens]
//     - Runs greedy single-stream generation: pre-fills the prompt (as raw text using the chat template
//       or as a pre-tokenized .npy of ids), then streams the detokenized text to stdout until EOS or
//       max_tokens.
//
// <dir>/<model> is either a local model directory or a HuggingFace repo id (e.g. mlx-community/Llama-3.2-1B-Instruct-4bit),
// which will be downloaded on first use.

#include <cstdio>
#include <string>
#include <vector>

#include "mlx/mlx.h"

#include "core/config.h"
#include "core/logging.h"
#include "core/model_source.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/single_stream.h"
#include "tokenizer/tokenizer.h"

// Alias for convenience
namespace mx = mlx::core;

// Utility function: Checks if string s ends with the given suffix.
namespace {
bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size()
    && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

namespace {

// build smoke test: verifies MLX graph building/eval and prints result
int run_smoke() {
  // Check if Metal GPU backend is available and log; return error otherwise
  if (!mx::metal::is_available()) {
    mlxforge::log::error("Metal GPU is not available on this machine");
    return 1;
  }
  mlxforge::log::info("Metal available: yes");

  // Create two arrays with the same shape
  mx::array a({1.0f, 2.0f, 3.0f, 4.0f});
  mx::array b({10.0f, 20.0f, 30.0f, 40.0f});
  // Add the arrays (creates a lazy operation graph)
  mx::array c = mx::add(a, b);

  // Evaluate the computation (forces the lazy graph to run)
  mx::eval(c);

  // Print the resulting array
  const float* data = c.data<float>();
  std::printf("a + b = [");
  for (int i = 0; i < c.size(); ++i) {
    std::printf("%g%s", data[i], i + 1 < c.size() ? ", " : "");
  }
  std::printf("]\n");
  return 0;
}

// Loads all weights from a model directory and checks their dtypes.
int run_dump_weights(const std::string& spec) {
  // Resolve the directory for the model, possibly downloading it
  const std::string dir = mlxforge::resolve_model_dir(spec);
  // Reset the recorded peak device memory for accurate measurement
  mx::reset_peak_memory();
  // Actually load all weight tensors from the directory
  mlxforge::Weights w = mlxforge::load_weights(dir);

  // Materialize (evaluate) all tensors so memory usage is valid,
  // and enforce that any fp16 cast is done before measurement
  std::vector<mx::array> all;
  all.reserve(w.tensors.size());
  for (auto& [_, a] : w.tensors)
    all.push_back(a);
  mx::eval(all);

  // Print a summary of all tensors: key, shape, dtype, etc.
  std::printf("%s", w.summary().c_str());

  // Check if all tensors are fp16, and count the ones that are not
  size_t non_fp16 = 0;
  for (const auto& [_, a] : w.tensors) {
    if (a.dtype() != mx::float16)
      ++non_fp16;
  }

  // Calculate resident peak memory in GiB
  const double gib = static_cast<double>(mx::get_peak_memory()) / (1024.0 * 1024.0 * 1024.0);

  // Log statistics on weight tensors and memory usage.
  mlxforge::log::info("{} tensors loaded; {} non-fp16; peak memory {:.2f} GiB",
                      w.size(), non_fp16, gib);
  if (non_fp16 > 0)
    mlxforge::log::warn("{} tensors are not fp16", non_fp16);

  // Return nonzero if some tensors are not fp16, zero otherwise.
  return non_fp16 == 0 ? 0 : 1;
}

// Performs generation using a loaded model, with either raw text or pre-tokenized prompts.
int run_generate(const std::string& spec, const std::string& prompt_arg, int max_tokens) {
  // Resolve and load the model directory (downloading if needed)
  const std::string dir = mlxforge::resolve_model_dir(spec);

  // Load the model configuration from file
  mlxforge::ModelConfig cfg = mlxforge::ModelConfig::from_file(dir + "/config.json");

  // Load the model weights and construct the LlamaModel
  mlxforge::LlamaModel model(cfg, mlxforge::load_weights(dir));

  // Load the tokenizer with proper configuration
  mlxforge::Tokenizer tok = mlxforge::Tokenizer::from_file(
      dir + "/tokenizer.json", cfg.bos_token_id,
      mlxforge::chat_format_from_model_type(cfg.model_type)
  );

  // Prepare the prompt, either by loading a .npy of pre-tokenized ids
  // or by applying the chat template to the provided raw prompt text.
  std::vector<int> prompt;
  if (ends_with(prompt_arg, ".npy")) {
    // Pre-tokenized prompt: load as int32 and evaluate so CPU data is current
    mx::array ids = mx::contiguous(mx::astype(mx::load(prompt_arg), mx::int32));
    mx::eval(ids);
    prompt.assign(ids.data<int32_t>(), ids.data<int32_t>() + ids.size());
  } else {
    // Raw text: wrap as user message and run through chat template
    prompt = tok.apply_chat_template({{"user", prompt_arg}});
  }

  // Create a streaming detokenizer for outputting human-readable text incrementally
  mlxforge::StreamingDetokenizer detok(tok);

  // Call greedy_generate, providing a lambda which receives each token id
  mlxforge::GenerateResult r =
      mlxforge::greedy_generate(model, prompt, max_tokens, cfg.eos_token_ids,
        [&](int id) {
          // For every generated token: detokenize and stream to stdout
          std::string piece = detok.add(id);
          std::fwrite(piece.data(), 1, piece.size(), stdout);
          std::fflush(stdout);
       });

  // Output any final detokenized tail remaining in the streaming detokenizer
  std::string tail = detok.finish();
  std::fwrite(tail.data(), 1, tail.size(), stdout);
  std::fputc('\n', stdout);

  // Log some generation statistics
  mlxforge::log::info("generated {} tokens{}", r.tokens.size(),
                      r.hit_eos ? " (stopped at EOS)" : "");
  mlxforge::log::info("time to first token {:.1f}ms; decode {:.1f} tok/s ({} tokens in {:.1f}ms)",
                      r.ttft_ms, r.decode_tokens_per_second(), r.decode_tokens, r.decode_ms);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize logging
  mlxforge::log::init();

  // Parse the command (first argument, or empty string if not provided)
  const std::string cmd = argc >= 2 ? argv[1] : "";
  if (cmd == "dump-weights") {
    // Print usage if not enough arguments; otherwise run weights dump logic
    if (argc < 3) {
      std::fprintf(stderr, "usage: mlxforge-cli dump-weights <model_dir>\n");
      return 2;
    }
    return run_dump_weights(argv[2]);
  }
  if (cmd == "generate") {
    // Print usage if not enough arguments; otherwise run generation logic
    if (argc < 4) {
      std::fprintf(stderr, "usage: mlxforge-cli generate <model_dir> <prompt_ids.npy> [max_tokens]\n");
      return 2;
    }
    // Parse max_tokens if provided, otherwise default to 64
    const int max_tokens = argc >= 5 ? std::stoi(argv[4]) : 64;
    return run_generate(argv[2], argv[3], max_tokens);
  }

  // No subcommand: run the smoke test by default
  return run_smoke();
}
