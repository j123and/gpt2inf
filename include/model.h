#pragma once
#include "tensor.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace gpt2 {

struct GPT2Config {
    int n_vocab = 50257;
    int n_ctx   = 1024;
    int n_embd  = 768;
    int n_head  = 12;
    int n_layer = 12;
    float layer_norm_eps = 1e-5f;
};

class GPT2Model {
public:
    bool load(const std::string& weights_dir);
    Tensor forward(const std::vector<int>& token_ids);
    const GPT2Config& config() const { return config_; }

    double time_embedding = 0;
    double time_attention = 0;
    double time_mlp = 0;
    double time_layernorm = 0;
    double time_logits = 0;

private:
    bool load_config(const std::string& path);
    bool load_weights(const std::string& path);
    void attention(int layer, const Tensor& x, Tensor& out, int seq_len);
    void mlp(int layer, const Tensor& x, Tensor& out, int seq_len);

    GPT2Config config_;
    std::unordered_map<std::string, Tensor> weights_;
};

} // namespace gpt2