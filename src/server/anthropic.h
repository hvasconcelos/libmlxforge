// Anthropic Messages API parsing + response serialization (pure functions, no
// server/GPU — unit tested). An Anthropic request is normalized into the shared
// ChatRequest (server/openai.h); only the request shape and the response/SSE
// framing differ from the OpenAI surface, so the generation pipeline is reused.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/openai.h"

namespace mlxforge {

// Parse a POST /v1/messages body into the shared ChatRequest. The Anthropic
// shape differs from OpenAI: a top-level `system`, message `content` that is a
// string or an array of blocks (text / tool_use / tool_result), `max_tokens`
// required, `stop_sequences`, tool schemas under `input_schema`, and a
// structured `tool_choice`. Throws std::runtime_error on malformed shape or
// out-of-range params (the server maps that to a 400).
ChatRequest parse_messages_request(const nlohmann::json& body);

// Render parsed tool calls as Anthropic `tool_use` content blocks
// ({type, id:"toolu_N", name, input}). `input` is the call arguments object.
nlohmann::json make_tool_use_blocks(const std::vector<ToolCall>& calls);

// Serialize a finished generation into the Anthropic Messages response shape:
// {id, type:"message", role:"assistant", content:[blocks], model, stop_reason,
//  stop_sequence:null, usage:{input_tokens, output_tokens}}. `content` is the
// already-built content-block array (a text block and/or tool_use blocks).
nlohmann::json make_message_response(const std::string& id, const std::string& model,
                                     const nlohmann::json& content,
                                     const std::string& stop_reason, int input_tokens,
                                     int output_tokens);

// An Anthropic error body: {type:"error", error:{type, message}}.
nlohmann::json anthropic_error_body(const std::string& type, const std::string& message);

// Frame a payload as a named SSE event: "event: <name>\ndata: <json>\n\n".
// Anthropic streaming uses named events, unlike the OpenAI data-only frames.
std::string sse_event(const std::string& name, const nlohmann::json& payload);

// Streaming event payload builders (each returns the `data:` JSON object). The
// stream is: message_start, then per content block
// content_block_start/_delta*/_stop, then message_delta (carrying the final
// stop_reason + output token count), then message_stop.
nlohmann::json make_message_start(const std::string& id, const std::string& model,
                                  int input_tokens);
nlohmann::json make_content_block_start(int index, const nlohmann::json& block);
nlohmann::json make_text_delta(int index, const std::string& text);
nlohmann::json make_input_json_delta(int index, const std::string& partial_json);
nlohmann::json make_content_block_stop(int index);
nlohmann::json make_message_delta(const std::string& stop_reason, int output_tokens);
extern const nlohmann::json kMessageStop;

}  // namespace mlxforge
