#pragma once
#include "tensor.h"

namespace gpt2 {
namespace ops {

// C[M,N] = A[M,K] @ B[K,N]
void matmul(const Tensor& A, const Tensor& B, Tensor& C);

// Layernorm over last dimension
void layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
               float eps, Tensor& out);

// Softmax over last dim, row by row for seq_len rows
void softmax(Tensor& x, int rows);

// In-place GELU
void gelu(Tensor& x);

// a += b (broadcasts b over leading dims if smaller)
void add_inplace(Tensor& a, const Tensor& b);

// Copy one row from embedding table
void embedding_lookup(int token_id, const Tensor& table, float* out, int d);

} // namespace ops
} // namespace gpt2