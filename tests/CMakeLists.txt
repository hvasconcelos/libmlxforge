# Test binaries. The suite is split by cost so a plain `ctest` does not reload
# the model checkpoints dozens of times:
#
#   mlxforge_tests              pure-logic units (no model weights). Registered
#                               per-TEST_CASE via doctest_discover_tests for
#                               granular reporting; each runs in its own process,
#                               which is cheap because nothing loads a model.
#   mlxforge_integration_tests  model-backed tests (forward pass, cache, runtime,
#                               scheduler, GGUF) for the small checkpoints. Run as
#                               a SINGLE ctest test so the whole binary executes
#                               in one process — the shared_*_model() statics load
#                               each checkpoint exactly once and reuse it across
#                               every case, instead of reloading per case.
#   mlxforge_moe_tests          the Qwen3 MoE (~17GB 30B-A3B) golden-reference
#                               tests, in their own single-process binary so that
#                               large checkpoint is loaded once and its peak
#                               memory is isolated from the other models.
#
# All three self-skip when their model isn't cached (so CI, which downloads no
# weights, runs only the pure-logic units + committed fixtures).
include(doctest)

# Pure-logic unit tests: no model weights, fast, granular ctest reporting.
add_executable(mlxforge_tests
  doctest_main.cpp
  smoke_test.cpp
  core/config_test.cpp
  core/weights_test.cpp
  core/model_source_test.cpp
  support/reference_test.cpp
  cache/batch_kv_cache_test.cpp
  cache/batch_kv_filter_merge_test.cpp
  cache/kv_budget_test.cpp
  sample/sampler_test.cpp
  scheduler/scheduler_test.cpp
  tokenizer/tokenizer_test.cpp
  tokenizer/bpe_test.cpp
  server/openai_test.cpp
  server/anthropic_test.cpp
  server/sse_test.cpp
  server/hardening_test.cpp
)

# Model-backed integration tests for the small checkpoints (Llama 1B fp16/4bit,
# Qwen3 0.6B, Llama GGUF). One binary -> one process -> each model loads once.
add_executable(mlxforge_integration_tests
  doctest_main.cpp
  cache/kv_cache_test.cpp
  core/gguf_test.cpp
  model/llama_front_test.cpp
  model/llama_block_test.cpp
  model/llama_forward_test.cpp
  model/batched_decode_test.cpp
  model/quantized_test.cpp
  model/qwen3_test.cpp
  runtime/single_stream_test.cpp
  runtime/prefill_test.cpp
  runtime/bucketing_test.cpp
  scheduler/continuous_batch_test.cpp
  scheduler/worker_test.cpp
  scheduler/validation_test.cpp
)

# Qwen3 MoE golden-reference tests, isolated so the large checkpoint loads once
# and does not coexist in memory with the other models.
add_executable(mlxforge_moe_tests
  doctest_main.cpp
  model/qwen3_moe_test.cpp
)

# Resolve the local HF snapshot dir for integration tests (empty -> tests skip).
file(GLOB _mlxforge_model_snapshots
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-bf16/snapshots/*")
file(GLOB _mlxforge_model_snapshots_4bit
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/*")
set(MLXFORGE_MODEL_DIR_4BIT "")
if(_mlxforge_model_snapshots_4bit)
  list(GET _mlxforge_model_snapshots_4bit 0 MLXFORGE_MODEL_DIR_4BIT)
endif()
set(MLXFORGE_MODEL_DIR "")
if(_mlxforge_model_snapshots)
  list(GET _mlxforge_model_snapshots 0 MLXFORGE_MODEL_DIR)
endif()

# Qwen3 dense model snapshot for the QK-Norm / ChatML integration tests (empty
# -> those tests self-skip).
file(GLOB _mlxforge_model_snapshots_qwen3
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/*")
set(MLXFORGE_MODEL_DIR_QWEN3 "")
if(_mlxforge_model_snapshots_qwen3)
  list(GET _mlxforge_model_snapshots_qwen3 0 MLXFORGE_MODEL_DIR_QWEN3)
endif()

# Qwen3 MoE model snapshot for the sparse-expert integration tests (empty -> those
# tests self-skip). The 30B-A3B-4bit checkpoint is large (~16GB); absent is fine.
file(GLOB _mlxforge_model_snapshots_qwen3_moe
  "$ENV{HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit/snapshots/*")
set(MLXFORGE_MODEL_DIR_QWEN3_MOE "")
if(_mlxforge_model_snapshots_qwen3_moe)
  list(GET _mlxforge_model_snapshots_qwen3_moe 0 MLXFORGE_MODEL_DIR_QWEN3_MOE)
endif()

# A local Llama GGUF file for the self-contained-loader integration tests
# (empty -> the GGUF tests self-skip). Any *.gguf under the cached bartowski
# Llama-3.2-1B-Instruct-GGUF repo is used.
file(GLOB _mlxforge_gguf_files
  "$ENV{HOME}/.cache/huggingface/hub/models--bartowski--Llama-3.2-1B-Instruct-GGUF/snapshots/*/*.gguf")
set(MLXFORGE_GGUF_MODEL "")
if(_mlxforge_gguf_files)
  list(GET _mlxforge_gguf_files 0 MLXFORGE_GGUF_MODEL)
endif()

# Shared wiring for every test binary: include path, links, and the fixture /
# model-dir compile definitions (the binaries that don't touch a given model
# simply ignore the corresponding define).
set(_mlxforge_test_targets mlxforge_tests mlxforge_integration_tests mlxforge_moe_tests)
foreach(_t IN LISTS _mlxforge_test_targets)
  target_include_directories(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${_t} PRIVATE mlxforge_core doctest::doctest)
  target_compile_definitions(${_t} PRIVATE
    MLXFORGE_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    MLXFORGE_REF_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/reference/fixtures"
    MLXFORGE_REF_FIXTURES_DIR_QWEN3="${CMAKE_SOURCE_DIR}/reference/fixtures_qwen3"
    MLXFORGE_REF_FIXTURES_DIR_QWEN3_MOE="${CMAKE_SOURCE_DIR}/reference/fixtures_qwen3_moe"
    MLXFORGE_MODEL_DIR="${MLXFORGE_MODEL_DIR}"
    MLXFORGE_MODEL_DIR_4BIT="${MLXFORGE_MODEL_DIR_4BIT}"
    MLXFORGE_MODEL_DIR_QWEN3="${MLXFORGE_MODEL_DIR_QWEN3}"
    MLXFORGE_MODEL_DIR_QWEN3_MOE="${MLXFORGE_MODEL_DIR_QWEN3_MOE}"
    MLXFORGE_GGUF_MODEL="${MLXFORGE_GGUF_MODEL}")
endforeach()

# Units: one ctest entry per TEST_CASE (cheap — no model loads).
doctest_discover_tests(mlxforge_tests)

# Integration + MoE: one ctest entry each, so the whole binary runs in a single
# process and the shared model statics load each checkpoint once. The generous
# timeout covers a cold run that also has to load weights from disk.
add_test(NAME integration COMMAND mlxforge_integration_tests)
add_test(NAME integration_moe COMMAND mlxforge_moe_tests)
set_tests_properties(integration integration_moe PROPERTIES TIMEOUT 3600)
