#include "ops.h"
#include <cmath>
#include <algorithm>

namespace gpt2 {
namespace ops {

// void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
//     int M = A.shape(0), K = A.shape(1), N = B.shape(1);
//     C.zero();
//     for (int i = 0; i < M; i++)
//         for (int j = 0; j < N; j++) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; k++)
//                 sum += A.at(i, k) * B.at(k, j);
//             C.at(i, j) = sum;
//         }
// }

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shape(0), K = A.shape(1), N = B.shape(1);
    C.zero();
    #pragma omp parallel for
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a = A.at(i, k);
            for (int j = 0; j < N; j++)
                C.at(i, j) += a * B.at(k, j);
        }
}

void layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
               float eps, Tensor& out) {
    int d = x.shape(-1);
    int rows = x.numel() / d;
    for (int i = 0; i < rows; i++) {
        const float* row = x.data() + i * d;
        float* dst = out.data() + i * d;

        float mean = 0.0f;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;

        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= d;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < d; j++)
            dst[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

void softmax(Tensor& x, int rows) {
    int cols = x.shape(-1);
    for (int i = 0; i < rows; i++) {
        float* row = x.data() + i * cols;
        float max_val = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }
        for (int j = 0; j < cols; j++) row[j] /= sum;
    }
}

void gelu(Tensor& x) {
    constexpr float c = 0.7978845608f; // sqrt(2/pi)
    for (int i = 0; i < x.numel(); i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.numel() == b.numel()) {
        for (int i = 0; i < a.numel(); i++) a[i] += b[i];
        return;
    }
    int d = b.numel();
    int rows = a.numel() / d;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < d; j++)
            a[i * d + j] += b[j];
}

void embedding_lookup(int token_id, const Tensor& table, float* out, int d) {
    const float* src = table.data() + token_id * d;
    std::copy(src, src + d, out);
}

} // namespace ops
} // namespace gpt2