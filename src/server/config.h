// XLLM-024: server configuration from CLI flags / environment.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace xllm {

struct ServerConfig {
  std::string model_dir;
  std::string host = "0.0.0.0";
  int port = 8080;
  int max_ctx = 8192;               // reject prompts longer than this -> 400
  int max_waiting = 256;            // bounded waiting queue -> 429 on overflow
  std::size_t kv_budget_bytes = 0;  // 0 = unbounded (XLLM-012 gate)

  // Parse argv (positional model_dir, then --flag value / --flag=value) with
  // environment fallback (XLLM_HOST, XLLM_PORT, XLLM_MAX_CTX, XLLM_MAX_WAITING,
  // XLLM_KV_BUDGET). Throws std::runtime_error on an unknown/!malformed flag.
  static ServerConfig parse(const std::vector<std::string>& args);
};

}  // namespace xllm
