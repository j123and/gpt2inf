#!/usr/bin/env python3
"""Download GPT-2 124M and export weights to binary format."""

import argparse
import json
import struct
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAGIC = 0x67707432  # "gpt2"

def export_weights(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading GPT-2 124M...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config = model.config

    # --- Save config.json ---
    cfg = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "layer_norm_eps": config.layer_norm_epsilon,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved config.json: {cfg}")

    # --- Save vocab.json and merges.txt ---
    tokenizer.save_vocabulary(output_dir)
    print(f"Saved vocab.json and merges.txt")

    # --- Export weights to binary ---
    state_dict = model.state_dict()

    # GPT-2 ties wte and lm_head. We'll store these weight names:
    #   wte.weight, wpe.weight
    #   h.{i}.ln_1.weight, h.{i}.ln_1.bias
    #   h.{i}.attn.c_attn.weight, h.{i}.attn.c_attn.bias
    #   h.{i}.attn.c_proj.weight, h.{i}.attn.c_proj.bias
    #   h.{i}.ln_2.weight, h.{i}.ln_2.bias
    #   h.{i}.mlp.c_fc.weight, h.{i}.mlp.c_fc.bias
    #   h.{i}.mlp.c_proj.weight, h.{i}.mlp.c_proj.bias
    #   ln_f.weight, ln_f.bias

    # Rename keys from HuggingFace format to our flat names
    rename = {}
    rename["transformer.wte.weight"] = "wte.weight"
    rename["transformer.wpe.weight"] = "wpe.weight"
    rename["transformer.ln_f.weight"] = "ln_f.weight"
    rename["transformer.ln_f.bias"] = "ln_f.bias"

    for i in range(config.n_layer):
        hf = f"transformer.h.{i}"
        ours = f"h.{i}"
        rename[f"{hf}.ln_1.weight"] = f"{ours}.ln_1.weight"
        rename[f"{hf}.ln_1.bias"] = f"{ours}.ln_1.bias"
        rename[f"{hf}.attn.c_attn.weight"] = f"{ours}.attn.c_attn.weight"
        rename[f"{hf}.attn.c_attn.bias"] = f"{ours}.attn.c_attn.bias"
        rename[f"{hf}.attn.c_proj.weight"] = f"{ours}.attn.c_proj.weight"
        rename[f"{hf}.attn.c_proj.bias"] = f"{ours}.attn.c_proj.bias"
        rename[f"{hf}.ln_2.weight"] = f"{ours}.ln_2.weight"
        rename[f"{hf}.ln_2.bias"] = f"{ours}.ln_2.bias"
        rename[f"{hf}.mlp.c_fc.weight"] = f"{ours}.mlp.c_fc.weight"
        rename[f"{hf}.mlp.c_fc.bias"] = f"{ours}.mlp.c_fc.bias"
        rename[f"{hf}.mlp.c_proj.weight"] = f"{ours}.mlp.c_proj.weight"
        rename[f"{hf}.mlp.c_proj.bias"] = f"{ours}.mlp.c_proj.bias"

    # HuggingFace GPT-2 Conv1D stores weights as [in, out] — we need [in, out]
    # for standard matmul x @ W where x is [seq, in] and W is [in, out].
    # Conv1D already stores them this way, so no transpose needed.

    tensors = {}
    for hf_name, our_name in rename.items():
        if hf_name not in state_dict:
            print(f"  WARNING: {hf_name} not found in state_dict")
            continue
        t = state_dict[hf_name].float().numpy()
        tensors[our_name] = t
        print(f"  {our_name}: {t.shape} ({t.dtype})")

    # Write binary file
    bin_path = os.path.join(output_dir, "gpt2_124m.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", len(tensors)))

        for name, arr in tensors.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            shape = arr.shape
            f.write(struct.pack("<I", len(shape)))
            for s in shape:
                f.write(struct.pack("<I", s))

            f.write(arr.astype(np.float32).tobytes())

    file_size = os.path.getsize(bin_path)
    print(f"\nSaved {bin_path} ({file_size / 1e6:.1f} MB, {len(tensors)} tensors)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/", help="Output directory")
    args = parser.parse_args()
    export_weights(args.output)