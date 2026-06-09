#include "model/qwen3_vl.h"

#include <algorithm>

#include "mlx/ops.h"

namespace mlxforge {

mx::array mrope_position_ids(const std::vector<int>& input_ids,
                             const std::vector<std::array<int, 3>>& image_grids,
                             const ModelConfig& cfg) {
  const int seq = static_cast<int>(input_ids.size());
  const int merge = cfg.vision ? cfg.vision->spatial_merge_size : 1;
  const int image_tok = cfg.image_token_id;

  std::array<std::vector<int>, 3> rows;  // temporal, height, width
  for (auto& r : rows) r.reserve(seq);

  int base = 0;       // next position to assign (== max so far + 1)
  size_t image_idx = 0;
  int p = 0;
  while (p < seq) {
    if (input_ids[p] == image_tok && image_idx < image_grids.size()) {
      // An image run: t * (h/merge) * (w/merge) collapsed patches, laid out in
      // (frame, height, width) order. Each axis is offset from the same base; the
      // base then jumps past the largest extent (so text resumes after the image).
      const auto& g = image_grids[image_idx++];
      const int t = g[0], gh = g[1] / merge, gw = g[2] / merge;
      for (int f = 0; f < t; ++f)
        for (int hh = 0; hh < gh; ++hh)
          for (int ww = 0; ww < gw; ++ww) {
            rows[0].push_back(base + f);
            rows[1].push_back(base + hh);
            rows[2].push_back(base + ww);
          }
      base += std::max({t, gh, gw});
      p += t * gh * gw;
    } else {
      // Text token: equal positions on all three axes.
      for (auto& r : rows) r.push_back(base);
      ++base;
      ++p;
    }
  }

  // Pack the three rows into a (3, seq) int32 array.
  std::vector<int> flat;
  flat.reserve(static_cast<size_t>(3) * seq);
  for (const auto& r : rows) flat.insert(flat.end(), r.begin(), r.end());
  return mx::array(flat.data(), {3, seq}, mx::int32);
}

}  // namespace mlxforge
