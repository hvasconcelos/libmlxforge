// Self-contained HTML renderer for the CLI `schematic` command.
//
// Takes a ModelSchema JSON blob (model_schema.h's to_json) and splices it into
// an embedded HTML template — inline CSS/JS/SVG only, no external references —
// so the output is a single file that renders offline. The page draws the
// transformer-block schematic, stat cards, parameter-distribution bars and a
// collapsible per-layer tensor explorer with vanilla JS.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace mlxforge::inspect {

std::string render_schematic_html(const nlohmann::json& schema);

}  // namespace mlxforge::inspect
