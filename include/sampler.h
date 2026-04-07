#pragma once
#include <algorithm>
#include <random>

namespace gpt2 {

class Sampler {
public:
    explicit Sampler(float temperature = 1.0f, uint64_t seed = 42)
        : temperature_(temperature), rng_(seed) {}

    int greedy(const float* logits, int vocab_size) {
        return static_cast<int>(
            std::max_element(logits, logits + vocab_size) - logits);
    }

private:
    float temperature_;
    std::mt19937 rng_;
};

} // namespace gpt2