#include "cache/batch_kv_cache.h"

#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace xllm {

namespace {
// offset starts at -left_padding (per row).
mx::array neg(const std::vector<int>& v) {
  std::vector<int> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = -v[i];
  return mx::array(out.data(), {static_cast<int>(out.size())}, mx::int32);
}
}  // namespace

BatchKVCache::BatchKVCache(int n_layers, const std::vector<int>& left_padding)
    : batch_(static_cast<int>(left_padding.size())),
      keys_(n_layers),
      values_(n_layers),
      offset_(neg(left_padding)),
      left_padding_(mx::array(left_padding.data(), {static_cast<int>(left_padding.size())},
                              mx::int32)) {}

int BatchKVCache::s_cap() const {
  return keys_[0].has_value() ? keys_[0]->shape()[2] : 0;
}

std::pair<mx::array, mx::array> BatchKVCache::update_and_fetch(int layer, const mx::array& k,
                                                              const mx::array& v) {
  const int prev = idx_;
  const int L = k.shape()[2];
  const int end = prev + L;
  const int B = k.shape()[0];
  const int H = k.shape()[1];
  const int Dk = k.shape()[3];
  const int Dv = v.shape()[3];

  const int cap = keys_[layer].has_value() ? keys_[layer]->shape()[2] : 0;
  if (!keys_[layer].has_value() || end > cap) {
    const int n_steps = (kStep + L - 1) / kStep;
    const int add = n_steps * kStep;
    mx::array new_k = mx::zeros({B, H, add, Dk}, k.dtype());
    mx::array new_v = mx::zeros({B, H, add, Dv}, v.dtype());
    if (keys_[layer].has_value()) {
      mx::array kk = *keys_[layer];
      mx::array vv = *values_[layer];
      // Drop any unused tail of the last block before growing.
      if (prev % kStep != 0) {
        kk = mx::slice(kk, {0, 0, 0, 0}, {B, H, prev, Dk});
        vv = mx::slice(vv, {0, 0, 0, 0}, {B, H, prev, Dv});
      }
      keys_[layer] = mx::concatenate({kk, new_k}, /*axis=*/2);
      values_[layer] = mx::concatenate({vv, new_v}, /*axis=*/2);
    } else {
      keys_[layer] = new_k;
      values_[layer] = new_v;
    }
  }

  keys_[layer] = mx::slice_update(*keys_[layer], k, {0, 0, prev, 0}, {B, H, end, Dk});
  values_[layer] = mx::slice_update(*values_[layer], v, {0, 0, prev, 0}, {B, H, end, Dv});
  return {mx::slice(*keys_[layer], {0, 0, 0, 0}, {B, H, end, Dk}),
          mx::slice(*values_[layer], {0, 0, 0, 0}, {B, H, end, Dv})};
}

void BatchKVCache::advance(int n_tokens) {
  idx_ += n_tokens;
  offset_ = mx::add(offset_, mx::array(n_tokens, mx::int32));
  mx::eval(offset_);  // keep the per-row bookkeeping materialized
}

}  // namespace xllm
