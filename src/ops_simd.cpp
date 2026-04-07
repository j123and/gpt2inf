#include "ops.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace gpt2 {
namespace ops {

void matmul(const Tensor& A, const Tensor& B, Tensor& C) {
    int M = A.shape(0), K = A.shape(1), N = B.shape(1);
    C.zero();

    const float* a_data = A.data();
    const float* b_data = B.data();
    float* c_data = C.data();

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            __m256 a_vec = _mm256_set1_ps(a_data[i * K + k]);
            const float* b_row = b_data + k * N;
            float* c_row = c_data + i * N;

            int j = 0;
            for (; j + 7 < N; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(b_row + j);
                __m256 c_vec = _mm256_loadu_ps(c_row + j);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(c_row + j, c_vec);
            }
            float a_val = a_data[i * K + k];
            for (; j < N; j++)
                c_row[j] += a_val * b_row[j];
        }
    }
}

void layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
               float eps, Tensor& out) {
    int d = x.shape(-1);
    int rows = x.numel() / d;

    for (int i = 0; i < rows; i++) {
        const float* row = x.data() + i * d;
        float* dst = out.data() + i * d;

        // Vectorized mean
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < d; j += 8)
            sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(row + j));
        // Horizontal sum of the 8 lanes
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float mean = _mm_cvtss_f32(s);
        for (; j < d; j++) mean += row[j];
        mean /= d;

        // Vectorized variance
        __m256 mean_vec = _mm256_set1_ps(mean);
        __m256 var_vec = _mm256_setzero_ps();
        j = 0;
        for (; j + 7 < d; j += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(row + j), mean_vec);
            var_vec = _mm256_fmadd_ps(diff, diff, var_vec);
        }
        hi = _mm256_extractf128_ps(var_vec, 1);
        lo = _mm256_castps256_ps128(var_vec);
        s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float var = _mm_cvtss_f32(s);
        for (; j < d; j++) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= d;

        // Vectorized normalize
        __m256 inv_std_vec = _mm256_set1_ps(1.0f / std::sqrt(var + eps));
        j = 0;
        for (; j + 7 < d; j += 8) {
            __m256 x_vec = _mm256_loadu_ps(row + j);
            __m256 g_vec = _mm256_loadu_ps(gamma.data() + j);
            __m256 b_vec = _mm256_loadu_ps(beta.data() + j);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), inv_std_vec);
            _mm256_storeu_ps(dst + j, _mm256_fmadd_ps(norm, g_vec, b_vec));
        }
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (; j < d; j++)
            dst[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

void softmax(Tensor& x, int rows) {
    int cols = x.shape(-1);
    for (int i = 0; i < rows; i++) {
        float* row = x.data() + i * cols;
        float max_val = *std::max_element(row, row + cols);

        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < cols; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(row + j), max_vec);
            // exp approximation: use scalar for correctness
            float tmp[8];
            _mm256_storeu_ps(tmp, v);
            for (int t = 0; t < 8; t++) tmp[t] = std::exp(tmp[t]);
            __m256 exp_v = _mm256_loadu_ps(tmp);
            _mm256_storeu_ps(row + j, exp_v);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
        }
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; j < cols; j++) {
            row[j] = std::exp(row[j] - max_val);
            sum += row[j];
        }

        __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
        j = 0;
        for (; j + 7 < cols; j += 8)
            _mm256_storeu_ps(row + j,
                _mm256_mul_ps(_mm256_loadu_ps(row + j), inv_sum));
        for (; j < cols; j++) row[j] /= sum;
    }
}

void gelu(Tensor& x) {
    constexpr float c = 0.7978845608f;
    int n = x.numel();
    int i = 0;
    for (; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.numel() == b.numel()) {
        int i = 0;
        for (; i + 7 < a.numel(); i += 8) {
            __m256 av = _mm256_loadu_ps(a.data() + i);
            __m256 bv = _mm256_loadu_ps(b.data() + i);
            _mm256_storeu_ps(a.data() + i, _mm256_add_ps(av, bv));
        }
        for (; i < a.numel(); i++) a[i] += b[i];
        return;
    }
    int d = b.numel();
    int rows = a.numel() / d;
    for (int r = 0; r < rows; r++) {
        int i = 0;
        for (; i + 7 < d; i += 8) {
            __m256 av = _mm256_loadu_ps(a.data() + r * d + i);
            __m256 bv = _mm256_loadu_ps(b.data() + i);
            _mm256_storeu_ps(a.data() + r * d + i, _mm256_add_ps(av, bv));
        }
        for (; i < d; i++) a[r * d + i] += b[i];
    }
}

void embedding_lookup(int token_id, const Tensor& table, float* out, int d) {
    const float* src = table.data() + token_id * d;
    std::copy(src, src + d, out);
}

} // namespace ops
} // namespace gpt2