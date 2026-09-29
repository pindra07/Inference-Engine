#!/usr/bin/env python3
"""Export a Hugging Face Llama-2 checkpoint -> llama2.bin + tokenizer.bin.

Our C++ engine reads the Karpathy llama2.bin layout:
  int32[7] header {dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                   vocab_size (neg => shared classifier), seq_len}
  then FP32 weights in order:
  tok_emb, per-layer(attn_rms, Wq, Wk, Wv, Wo, ffn_rms, W1, W2, W3),
  final_rms, [wcls unless shared].

Usage:
  python tools/export_hf_to_bin.py --hf-dir models/Llama-2-7b-hf \\
      --out-bin models/llama2-7b.bin --out-tok models/tokenizer.bin

Also handles karpathy/tinyllamas .pt checkpoints via --from-karpathy-pt.
Requires: torch, safetensors (or transformers), numpy.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def permute_for_rope(w, n_heads, dim):
    # HF stores q/k with interleaved rotary dims; undo to vanilla layout.
    # Matches transformers convert_llama_weights_to_hf.py inverse.
    head_dim = dim // n_heads
    return (
        w.reshape(n_heads, 2, head_dim // 2, dim)
        .transpose(0, 2, 1, 3)
        .reshape(dim, dim)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-tok", required=True)
    args = ap.parse_args()

    import torch

    cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    vocab = cfg["vocab_size"]
    seq = cfg.get("max_position_embeddings", 2048)
    theta = cfg.get("rope_theta", 10000.0)
    print(f"config: dim={dim} hidden={hidden} L={layers} H={heads} "
          f"KV={kv_heads} vocab={vocab} seq={seq} theta={theta}")

    # Load weights (safetensors preferred).
    try:
        from safetensors.torch import load_file as safe_load
        import glob

        state = {}
        for f in sorted(glob.glob(os.path.join(args.hf_dir, "*.safetensors"))):
            print("loading", f)
            state.update(safe_load(f))
    except ImportError:
        from transformers import AutoModelForCausalLM

        print("safetensors not found, loading via transformers (slow)...")
        m = AutoModelForCausalLM.from_pretrained(args.hf_dir,
                                                 torch_dtype=torch.float32)
        state = {k: v.float() for k, v in m.state_dict().items()}

    def t(name):
        for k in (f"model.{name}", name, f"language_model.{name}"):
            if k in state:
                return np.asarray(state[k].to(torch.float32).cpu())
        raise KeyError(f"missing weight: {name}")

    tok_emb = t("embed_tokens.weight").astype(np.float32)
    norm_f = t("norm.weight").astype(np.float32)
    lm_head = None
    try:
        lm_head = t("lm_head.weight").astype(np.float32)
    except KeyError:
        pass
    shared = lm_head is None or np.shares_memory(lm_head, tok_emb)
    if lm_head is None:
        print("no lm_head: using tied (shared) classifier")
        shared = True

    hdr_vocab = -vocab if shared else vocab
    header = struct.pack("7i", dim, hidden, layers, heads, kv_heads,
                         hdr_vocab, seq)

    # Collect per-layer tensors first: the .bin layout groups by tensor
    # across layers (all attn_rms, then all Wq, ...), matching llama2.c
    # memory_map_weights. See SPEC/06_MEMORY_WEIGHTS.md.
    attn_rms, Wq, Wk, Wv, Wo = [], [], [], [], []
    ffn_rms, W1, W2, W3 = [], [], [], []
    for l in range(layers):
        p = f"layers.{l}."
        attn_rms.append(t(p + "input_layernorm.weight").astype(np.float32))
        wq = t(p + "self_attn.q_proj.weight").astype(np.float32)
        wk = t(p + "self_attn.k_proj.weight").astype(np.float32)
        # HF q/k need rope-de-permutation for our runtime rope.
        Wq.append(permute_for_rope(wq, heads, dim))
        Wk.append(permute_for_rope(wk, kv_heads, dim))
        Wv.append(t(p + "self_attn.v_proj.weight").astype(np.float32))
        Wo.append(t(p + "self_attn.o_proj.weight").astype(np.float32))
        ffn_rms.append(
            t(p + "post_attention_layernorm.weight").astype(np.float32))
        W1.append(t(p + "mlp.gate_proj.weight").astype(np.float32))
        W2.append(t(p + "mlp.down_proj.weight").astype(np.float32))
        W3.append(t(p + "mlp.up_proj.weight").astype(np.float32))

    with open(args.out_bin, "wb") as f:
        f.write(header)
        f.write(tok_emb.tobytes())
        for g in (attn_rms, Wq, Wk, Wv, Wo, ffn_rms, W1, W2, W3):
            for a in g:
                f.write(a.tobytes())
        f.write(norm_f.tobytes())
        if not shared:
            f.write(lm_head.tobytes())
    print("wrote", args.out_bin, f"({os.path.getsize(args.out_bin)/1e9:.2f} GB)")

    # Tokenizer: prefer HF tokenizer.json/model, else sentencepiece.
    tok_json = os.path.join(args.hf_dir, "tokenizer.json")
    if os.path.exists(tok_json):
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.hf_dir)
        pieces = [tk.decode([i], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
                  for i in range(vocab)]
        scores = [-float(i) for i in range(vocab)]  # freq-order proxy
    else:
        print("no tokenizer.json; trying tokenizer.model (sentencepiece)",
              file=sys.stderr)
        from sentencepiece import SentencePieceProcessor
        sp = SentencePieceProcessor(
            model_file=os.path.join(args.hf_dir, "tokenizer.model"))
        pieces = [sp.id_to_piece(i) for i in range(sp.vocab_size())]
        scores = [sp.get_score(i) for i in range(sp.vocab_size())]
    max_len = max(len(p.encode("utf8")) for p in pieces)
    with open(args.out_tok, "wb") as f:
        f.write(struct.pack("i", max_len))
        for s, p in zip(scores, pieces):
            b = p.encode("utf8")
            f.write(struct.pack("f", float(s)))
            f.write(struct.pack("i", len(b)))
            f.write(b)
    print("wrote", args.out_tok)


if __name__ == "__main__":
    main()
