#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "timer.h"
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]) {
    std::string prompt = "The future of";
    int n_tokens = 5;
    std::string weights_dir = "weights/";

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i+1 < argc) n_tokens = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--weights") && i+1 < argc) weights_dir = argv[++i];
    }

    gpt2::GPT2Model model;
    if (!model.load(weights_dir)) return 1;

    gpt2::Tokenizer tokenizer;
    if (!tokenizer.load(weights_dir)) return 1;

    gpt2::Sampler sampler;

    auto token_ids = tokenizer.encode(prompt);
    std::cout << "\n" << prompt;

    gpt2::Timer total, t;
    double time_forward = 0, time_sample = 0, time_decode = 0;

    total.start();
    for (int i = 0; i < n_tokens; i++) {
        t.start();
        auto logits = model.forward(token_ids);
        time_forward += t.elapsed_ms();

        t.start();
        int next = sampler.greedy(logits.data(), model.config().n_vocab);
        time_sample += t.elapsed_ms();

        token_ids.push_back(next);

        t.start();
        std::cout << tokenizer.decode_token(next) << std::flush;
        time_decode += t.elapsed_ms();
    }

    double elapsed = total.elapsed_ms();
    std::cout << "\n\n=== Profile (" << n_tokens << " tokens) ===\n";
    std::cout << "Total:    " << elapsed << " ms\n";
    std::cout << "Forward:  " << time_forward << " ms ("
              << 100.0 * time_forward / elapsed << "%)\n";
    std::cout << "Sampling: " << time_sample << " ms ("
              << 100.0 * time_sample / elapsed << "%)\n";
    std::cout << "Decode:   " << time_decode << " ms ("
              << 100.0 * time_decode / elapsed << "%)\n";
    std::cout << "Tok/s:    " << 1000.0 * n_tokens / elapsed << "\n";
    std::cout << "\n=== Forward Breakdown ===\n";
    std::cout << "Embedding:  " << model.time_embedding << " ms\n";
    std::cout << "LayerNorm:  " << model.time_layernorm << " ms\n";
    std::cout << "Attention:  " << model.time_attention << " ms\n";
    std::cout << "MLP:        " << model.time_mlp << " ms\n";
    std::cout << "Logits:     " << model.time_logits << " ms\n";
    return 0;
}