// xllm — the OpenAI-compatible server binary (XLLM-022/023/024).
//   xllm <model_dir> [--host H] [--port P] [--max-ctx N] [--max-waiting N]
// Loads the tokenizer/config, starts the GPU worker, and serves the HTTP API.
// Config knobs also read from env (XLLM_HOST, XLLM_PORT, ...). SIGINT/SIGTERM
// trigger a graceful shutdown that drains in-flight requests.
#include <csignal>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/weights.h"
#include "model/llama.h"
#include "runtime/worker.h"
#include "scheduler/scheduler.h"
#include "server/config.h"
#include "server/http_server.h"
#include "tokenizer/tokenizer.h"

namespace {
xllm::HttpServer* g_server = nullptr;
void on_signal(int) {
  if (g_server) g_server->stop();  // unblocks listen(); main then drains
}
}  // namespace

int main(int argc, char** argv) {
  xllm::ServerConfig sc;
  try {
    sc = xllm::ServerConfig::parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "config error: %s\n", e.what());
    return 2;
  }
  if (sc.model_dir.empty()) {
    std::fprintf(stderr, "usage: xllm <model_dir> [--host H] [--port P] [--max-ctx N]\n");
    return 2;
  }

  xllm::ModelConfig cfg = xllm::ModelConfig::from_file(sc.model_dir + "/config.json");
  xllm::Tokenizer tok = xllm::Tokenizer::from_file(sc.model_dir + "/tokenizer.json");

  xllm::Scheduler scheduler;
  scheduler.set_max_waiting(sc.max_waiting);
  xllm::Worker worker(
      [dir = sc.model_dir] {
        return std::make_unique<xllm::LlamaModel>(
            xllm::ModelConfig::from_file(dir + "/config.json"), xllm::load_weights(dir));
      },
      &scheduler);
  worker.start();

  xllm::HttpServer server(&scheduler, &tok, cfg, "xllm", [&worker] { return worker.ready(); },
                          sc.max_ctx);
  g_server = &server;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  std::printf("xllm serving on http://%s:%d (max_ctx=%d max_waiting=%d)\n", sc.host.c_str(),
              sc.port, sc.max_ctx, sc.max_waiting);
  server.listen(sc.host, sc.port);  // blocks until stop()

  std::printf("draining in-flight requests...\n");
  worker.stop();  // drains the active batch before exit
  return 0;
}
