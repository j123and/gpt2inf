#include "model.h"
#include "ops.h"
#include "timer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>

namespace gpt2 {

static int json_get_int(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    pos = json.find(':', pos);
    return std::atoi(json.c_str() + pos + 1);
}

bool GPT2Model::load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    int v;
    v = json_get_int(json, "n_vocab");    if (v > 0) config_.n_vocab = v;
    v = json_get_int(json, "vocab_size"); if (v > 0) config_.n_vocab = v;
    v = json_get_int(json, "n_ctx");      if (v > 0) config_.n_ctx = v;
    v = json_get_int(json, "n_positions"); if (v > 0) config_.n_ctx = v;
    v = json_get_int(json, "n_embd");     if (v > 0) config_.n_embd = v;
    v = json_get_int(json, "n_head");     if (v > 0) config_.n_head = v;
    v = json_get_int(json, "n_layer");    if (v > 0) config_.n_layer = v;
    return true;
}

bool GPT2Model::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x67707432) {
        std::cerr << "Bad magic number\n";
        return false;
    }

    uint32_t n_tensors;
    f.read(reinterpret_cast<char*>(&n_tensors), 4);
    std::cout << "Loading " << n_tensors << " tensors...\n";

    for (uint32_t t = 0; t < n_tensors; t++) {
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<int> shape(ndim);
        for (uint32_t i = 0; i < ndim; i++) {
            uint32_t s;
            f.read(reinterpret_cast<char*>(&s), 4);
            shape[i] = static_cast<int>(s);
        }

        int numel = 1;
        for (int s : shape) numel *= s;
        std::vector<float> data(numel);
        f.read(reinterpret_cast<char*>(data.data()), numel * sizeof(float));

        weights_[name] = Tensor(shape, std::move(data));
    }

    std::cout << "Loaded " << weights_.size() << " tensors.\n";
    return true;
}

bool GPT2Model::load(const std::string& weights_dir) {
    std::string dir = weights_dir;
    if (!dir.empty() && dir.back() != '/') dir += '/';

    if (!load_config(dir + "config.json")) {
        std::cerr << "Could not load config.json\n";
        return false;
    }
    std::cout << "Config: vocab=" << config_.n_vocab
              << " embd=" << config_.n_embd
              << " heads=" << config_.n_head
              << " layers=" << config_.n_layer << "\n";

    if (!load_weights(dir + "gpt2_124m.bin")) {
        std::cerr << "Could not load weights\n";
        return false;
    }
    return true;
}

void GPT2Model::attention(int layer, const Tensor& x, Tensor& out, int seq_len) {
    int d = config_.n_embd;
    int n_head = config_.n_head;
    int d_head = d / n_head;
    std::string pfx = "h." + std::to_string(layer) + ".attn.";

    const Tensor& c_attn_w = weights_.at(pfx + "c_attn.weight");
    const Tensor& c_attn_b = weights_.at(pfx + "c_attn.bias");

    Tensor qkv({seq_len, 3 * d});
    ops::matmul(x, c_attn_w, qkv);
    ops::add_inplace(qkv, c_attn_b);

    out.zero();

    for (int h = 0; h < n_head; h++) {
        Tensor scores({seq_len, seq_len});
        float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (j > i) {
                    scores.at(i, j) = -1e9f;
                    continue;
                }
                float dot = 0.0f;
                for (int k = 0; k < d_head; k++) {
                    float q = qkv[i * 3 * d + h * d_head + k];
                    float kv = qkv[j * 3 * d + d + h * d_head + k];
                    dot += q * kv;
                }
                scores.at(i, j) = dot * scale;
            }
        }

        ops::softmax(scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int k = 0; k < d_head; k++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float v = qkv[j * 3 * d + 2 * d + h * d_head + k];
                    sum += scores.at(i, j) * v;
                }
                out[i * d + h * d_head + k] = sum;
            }
        }
    }

    const Tensor& c_proj_w = weights_.at(pfx + "c_proj.weight");
    const Tensor& c_proj_b = weights_.at(pfx + "c_proj.bias");
    Tensor proj_in = out;
    ops::matmul(proj_in, c_proj_w, out);
    ops::add_inplace(out, c_proj_b);
}

void GPT2Model::mlp(int layer, const Tensor& x, Tensor& out, int seq_len) {
    int d = config_.n_embd;
    int d_ff = 4 * d;
    std::string pfx = "h." + std::to_string(layer) + ".mlp.";

    const Tensor& fc_w = weights_.at(pfx + "c_fc.weight");
    const Tensor& fc_b = weights_.at(pfx + "c_fc.bias");
    Tensor hidden({seq_len, d_ff});
    ops::matmul(x, fc_w, hidden);
    ops::add_inplace(hidden, fc_b);
    ops::gelu(hidden);

    const Tensor& proj_w = weights_.at(pfx + "c_proj.weight");
    const Tensor& proj_b = weights_.at(pfx + "c_proj.bias");
    ops::matmul(hidden, proj_w, out);
    ops::add_inplace(out, proj_b);
}

Tensor GPT2Model::forward(const std::vector<int>& token_ids) {
    int seq_len = static_cast<int>(token_ids.size());
    int d = config_.n_embd;

    const Tensor& wte = weights_.at("wte.weight");
    const Tensor& wpe = weights_.at("wpe.weight");

    Timer t;

    t.start();
    Tensor x({seq_len, d});
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d; j++)
            x[i * d + j] = wte.at(token_ids[i], j) + wpe.at(i, j);
    time_embedding += t.elapsed_ms();

    Tensor ln_out({seq_len, d});
    Tensor attn_out({seq_len, d});
    Tensor mlp_out({seq_len, d});

    for (int l = 0; l < config_.n_layer; l++) {
        std::string pfx = "h." + std::to_string(l) + ".";

        t.start();
        ops::layernorm(x, weights_.at(pfx + "ln_1.weight"),
                       weights_.at(pfx + "ln_1.bias"),
                       config_.layer_norm_eps, ln_out);
        time_layernorm += t.elapsed_ms();

        t.start();
        attention(l, ln_out, attn_out, seq_len);
        time_attention += t.elapsed_ms();

        ops::add_inplace(x, attn_out);

        t.start();
        ops::layernorm(x, weights_.at(pfx + "ln_2.weight"),
                       weights_.at(pfx + "ln_2.bias"),
                       config_.layer_norm_eps, ln_out);
        time_layernorm += t.elapsed_ms();

        t.start();
        mlp(l, ln_out, mlp_out, seq_len);
        time_mlp += t.elapsed_ms();

        ops::add_inplace(x, mlp_out);
    }

    t.start();
    ops::layernorm(x, weights_.at("ln_f.weight"), weights_.at("ln_f.bias"),
                   config_.layer_norm_eps, ln_out);
    time_layernorm += t.elapsed_ms();

    t.start();
    Tensor logits({1, config_.n_vocab});
    const float* last = ln_out.data() + (seq_len - 1) * d;
    for (int v = 0; v < config_.n_vocab; v++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++)
            dot += last[j] * wte.at(v, j);
        logits[v] = dot;
    }
    time_logits += t.elapsed_ms();

    return logits;
}

} // namespace gpt2