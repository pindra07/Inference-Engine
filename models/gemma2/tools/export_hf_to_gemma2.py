#!/usr/bin/env python3
"""Export a Hugging Face Gemma2 checkpoint -> GGM2 .bin (see SPEC/10_GEMMA2.md).

GGM2 layout: magic "GGM2" + int32 version(1) + int32[9]
  {dim, hidden, layers, heads, kv_heads, head_dim, vocab, seq, sliding_window}
+ float32[5] {rope_theta, rms_eps, attn_scale, attn_softcap, final_softcap},
then FP32 weights GROUPED BY TENSOR:
  tok_emb, ALL input_rms, ALL Wq, ALL Wk, ALL Wv, ALL Wo, ALL post_attn_rms,
  ALL pre_ffn_rms, ALL Wgate, ALL Wdown, ALL Wup, ALL post_ffn_rms,
  final_rms, [lm_head iff untied].

Norms are exported RAW (folding (1+w) happens in the C++ loader).
q/k are de-permuted for vanilla RoPE (same as the llama2 exporter).

Usage:
  python models/gemma2/tools/export_hf_to_gemma2.py --hf-dir <hf> \
      --out-bin models/gemma2/weights/gemma2-2b.bin
Requires: torch, safetensors, numpy.
"""
import argparse
import glob
import json
import os
import struct

import numpy as np


def permute_for_rope(w, n_heads, head_dim, dim):
    # HF stores q/k heads-interleaved for rotary; undo to plain row-major.
    return (
        w.reshape(n_heads, 2, head_dim // 2, dim)
        .transpose(0, 2, 1, 3)
        .reshape(n_heads * head_dim, dim)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-bin", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.hf_dir, "config.json")))
    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim", dim // heads)
    vocab = cfg["vocab_size"]
    seq = cfg.get("max_position_embeddings", 8192)
    sw = cfg.get("sliding_window") or 0
    rope_cfg = cfg.get("rope_parameters") or {}
    theta = cfg.get("rope_theta", rope_cfg.get("rope_theta", 10000.0))
    eps = cfg.get("rms_norm_eps", 1e-6)
    qscale = cfg.get("query_pre_attn_scalar", 256)
    attn_cap = cfg.get("attn_logit_softcapping") or 0.0
    final_cap = cfg.get("final_logit_softcapping") or 0.0
    tied = cfg.get("tie_word_embeddings", True)
    if cfg.get("attention_bias", False):
        raise SystemExit("attention_bias=true not supported (SPEC/10)")
    assert heads * head_dim > 0 and kv_heads * head_dim > 0
    q_dim, kv_dim = heads * head_dim, kv_heads * head_dim
    print(f"config: dim={dim} hidden={hidden} L={layers} H={heads} "
          f"KV={kv_heads} hd={head_dim} vocab={vocab} seq={seq} sw={sw} "
          f"theta={theta} eps={eps} scale={qscale**-0.5:.6f} "
          f"caps=({attn_cap},{final_cap}) tied={tied}")

    from safetensors import safe_open

    state = {}
    files = sorted(glob.glob(os.path.join(args.hf_dir, "*.safetensors")))
    if not files:
        raise SystemExit("no .safetensors found in " + args.hf_dir)
    try:
        import torch  # noqa: F401  (for bf16/fp16 checkpoints)

        framework = "pt"
    except ImportError:
        framework = "np"
    for f in files:
        print("loading", f)
        with safe_open(f, framework=framework) as h:
            for k in h.keys():
                v = h.get_tensor(k)
                if framework == "pt":
                    v = v.to(torch.float32).cpu().numpy()
                state[k] = v

    def t(name):
        if name not in state:
            raise KeyError(f"missing weight: {name}")
        return np.asarray(state[name], dtype=np.float32)

    tok_emb = t("model.embed_tokens.weight")
    assert tok_emb.shape == (vocab, dim), tok_emb.shape
    norm_f = t("model.norm.weight")
    lm_head = state.get("lm_head.weight")
    if not tied and lm_head is None:
        raise SystemExit("untied but no lm_head found")
    if tied and lm_head is not None:
        print("note: lm_head present but tie_word_embeddings=true; tying")

    # Per-layer tensors (de-permute q/k for our runtime RoPE).
    groups = {k: [] for k in
              ("in", "wq", "wk", "wv", "wo", "post", "pre",
               "gate", "down", "up", "post_ffn")}
    for l in range(layers):
        p = f"model.layers.{l}."
        groups["in"].append(t(p + "input_layernorm.weight"))
        groups["wq"].append(
            permute_for_rope(t(p + "self_attn.q_proj.weight"), heads,
                             head_dim, dim))
        groups["wk"].append(
            permute_for_rope(t(p + "self_attn.k_proj.weight"), kv_heads,
                             head_dim, dim))
        groups["wv"].append(t(p + "self_attn.v_proj.weight"))
        groups["wo"].append(t(p + "self_attn.o_proj.weight"))
        groups["post"].append(t(p + "post_attention_layernorm.weight"))
        groups["pre"].append(t(p + "pre_feedforward_layernorm.weight"))
        groups["gate"].append(t(p + "mlp.gate_proj.weight"))
        groups["down"].append(t(p + "mlp.down_proj.weight"))
        groups["up"].append(t(p + "mlp.up_proj.weight"))
        groups["post_ffn"].append(t(p + "post_feedforward_layernorm.weight"))

    with open(args.out_bin, "wb") as f:
        f.write(struct.pack("<4s", b"GGM2"))
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<9i", dim, hidden, layers, heads, kv_heads,
                            head_dim, vocab, seq, sw))
        f.write(struct.pack("<5f", float(theta), float(eps),
                            float(qscale ** -0.5), float(attn_cap),
                            float(final_cap)))
        f.write(tok_emb.tobytes())
        for g in ("in", "wq", "wk", "wv", "wo", "post", "pre",
                  "gate", "down", "up", "post_ffn"):
            for a in groups[g]:
                f.write(a.tobytes())
        f.write(norm_f.tobytes())
        if not tied:
            f.write(np.asarray(lm_head, dtype=np.float32).tobytes())
    print("wrote", args.out_bin, f"({os.path.getsize(args.out_bin)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
