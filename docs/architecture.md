# GPT-2 124M Architecture

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50257 |
| Context length | 1024 |
| Embedding dim (d_model) | 768 |
| Attention heads | 12 |
| Head dim | 64 |
| Layers | 12 |
| MLP inner dim | 3072 (4 × 768) |

## Forward Pass

```
Input token IDs
       │
       ▼
┌─────────────────┐
│ Token Embedding  │  wte: [50257, 768]
│ + Position Emb   │  wpe: [1024, 768]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Transformer Block (×12)         │
│                                 │
│  ┌── LayerNorm 1 ──┐           │
│  │   Attention      │           │
│  │   Q,K,V proj     │ [768, 2304] (3 × 768)
│  │   Multi-head     │ 12 heads × 64 dim
│  │   Output proj    │ [768, 768]
│  └──── + residual ──┘           │
│                                 │
│  ┌── LayerNorm 2 ──┐           │
│  │   MLP            │           │
│  │   FC1            │ [768, 3072]
│  │   GELU           │
│  │   FC2            │ [3072, 768]
│  └──── + residual ──┘           │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Final LayerNorm  │  ln_f
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Logits (× wte^T)│  weight-tied with token embedding
└─────────────────┘
```

## Weight File Format (gpt2_124m.bin)

Binary file, little-endian:

```
[4 bytes] Magic number: 0x67707432 ("gpt2")
[4 bytes] Number of tensors (uint32)

For each tensor:
  [4 bytes] Name length in bytes (uint32)
  [N bytes] Name string (UTF-8, not null-terminated)
  [4 bytes] Number of dimensions (uint32)
  [4 bytes × ndim] Shape, each dimension as uint32
  [4 bytes × numel] Float32 data, row-major order
```

No alignment padding between tensors. Read sequentially with `fread`.

## Weight Names and Shapes

| Name | Shape | Description |
|------|-------|-------------|
| wte.weight | [50257, 768] | Token embeddings |
| wpe.weight | [1024, 768] | Position embeddings |
| h.{i}.ln_1.weight | [768] | Pre-attention layernorm γ |
| h.{i}.ln_1.bias | [768] | Pre-attention layernorm β |
| h.{i}.attn.c_attn.weight | [768, 2304] | QKV projection |
| h.{i}.attn.c_attn.bias | [2304] | QKV bias |
| h.{i}.attn.c_proj.weight | [768, 768] | Attention output projection |
| h.{i}.attn.c_proj.bias | [768] | Attention output bias |
| h.{i}.ln_2.weight | [768] | Pre-MLP layernorm γ |
| h.{i}.ln_2.bias | [768] | Pre-MLP layernorm β |
| h.{i}.mlp.c_fc.weight | [768, 3072] | MLP expansion |
| h.{i}.mlp.c_fc.bias | [3072] | MLP expansion bias |
| h.{i}.mlp.c_proj.weight | [3072, 768] | MLP projection |
| h.{i}.mlp.c_proj.bias | [768] | MLP projection bias |
| ln_f.weight | [768] | Final layernorm γ |
| ln_f.bias | [768] | Final layernorm β |

Total: 2 + 14 × 12 + 2 = **172 tensors**, ~497 MB as float32.