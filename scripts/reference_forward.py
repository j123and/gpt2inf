#!/usr/bin/env python3
"""Run GPT-2 forward pass and save intermediate activations for C++ validation."""

import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROMPT = "The future of"
OUTPUT_DIR = "reference/"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    # Tokenize
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
    tokens = input_ids[0].tolist()
    print(f"Prompt: \"{PROMPT}\"")
    print(f"Token IDs: {tokens}")

    # Save token IDs
    np.save(os.path.join(OUTPUT_DIR, "input_ids.npy"), np.array(tokens, dtype=np.int32))

    with torch.no_grad():
        # Use output_hidden_states to capture all intermediates cleanly
        outputs = model(input_ids, output_hidden_states=True)

        # hidden_states[0] = post-embedding
        # hidden_states[1] = post-layer-0
        # ...
        # hidden_states[12] = post-layer-11
        hs = outputs.hidden_states

        # Post-embedding
        post_emb = hs[0][0].numpy()  # [seq, 768]
        np.save(os.path.join(OUTPUT_DIR, "post_embedding.npy"), post_emb)
        print(f"Post-embedding: shape={post_emb.shape} first={post_emb[0, :5].tolist()}")

        # Post-layer-0
        post_l0 = hs[1][0].numpy()
        np.save(os.path.join(OUTPUT_DIR, "post_layer0.npy"), post_l0)
        print(f"Post-layer 0: shape={post_l0.shape} first={post_l0[0, :5].tolist()}")

        # Post-layer-11
        post_l11 = hs[12][0].numpy()
        np.save(os.path.join(OUTPUT_DIR, "post_layer11.npy"), post_l11)
        print(f"Post-layer 11: shape={post_l11.shape} first={post_l11[0, :5].tolist()}")

        # Final layernorm (apply manually since output_hidden_states gives pre-ln_f)
        t = model.transformer
        ln_f_out = t.ln_f(hs[12])[0].numpy()
        np.save(os.path.join(OUTPUT_DIR, "post_ln_f.npy"), ln_f_out)
        print(f"Post-ln_f: shape={ln_f_out.shape} first={ln_f_out[0, :5].tolist()}")

        # Logits
        logits = outputs.logits[0, -1].numpy()  # last token logits [50257]
        np.save(os.path.join(OUTPUT_DIR, "logits.npy"), logits)
        print(f"Logits shape: {logits.shape}")

        # Top-5 predictions
        logits_t = outputs.logits[0, -1]
        top5 = torch.topk(logits_t, 5)
        probs = torch.softmax(logits_t, dim=-1)
        print(f"\nTop-5 predictions:")
        for i in range(5):
            tid = top5.indices[i].item()
            token_str = tokenizer.decode([tid])
            print(f"  {i+1}. \"{token_str}\" (id={tid}, logit={top5.values[i].item():.4f}, prob={probs[tid].item():.4f})")

        # Greedy generate 20 tokens
        gen_ids = tokens.copy()
        for _ in range(20):
            out = model(torch.tensor([gen_ids]))
            next_id = out.logits[0, -1].argmax().item()
            gen_ids.append(next_id)
        print(f"\nGreedy 20 tokens: \"{tokenizer.decode(gen_ids)}\"")

    # Save summary
    summary = {
        "prompt": PROMPT,
        "token_ids": tokens,
        "files": [
            "input_ids.npy",
            "post_embedding.npy",
            "post_layer0.npy",
            "post_layer11.npy",
            "post_ln_f.npy",
            "logits.npy",
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(summary['files'])} activation files to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()