# 05 — Transformer: Llama-2 Math

Reference: Meta Llama-2 paper + `params.json` (7B: dim=4096, hidden=11008,
layers=32, heads=32, kv_heads=32, vocab=32000, rope_theta=10000).

## Per-layer forward (token t at position pos)

```
x = embedding[token]
for l in 0..n_layers-1:
  # attention block
  h = rmsnorm(x, attn_norm[l])                 # backend.rmsnorm
  q = Wq[l] @ h ; k = Wk[l] @ h ; v = Wv[l] @ h # backend.matvec ×3
  rope(q, k, pos)                              # backend.rope
  append k,v to cache[l][pos]
  a = attention(q, cache)                      # backend.attention (causal, GQA)
  o = Wo[l] @ a                                # backend.matvec
  x = x + o                                    # residual
  # FFN block (SwiGLU)
  h = rmsnorm(x, ffn_norm[l])
  g = W1[l] @ h ; u = W3[l] @ h                # backend.matvec ×2
  silu(g); g = g ⊙ u                           # backend.silu + matmul_add
  f = W2[l] @ g                                # backend.matvec
  x = x + f
logits = rmsnorm(x, final_norm) @ Wcls.T       # backend.rmsnorm + matvec
```

## RMSNorm
`out[i] = x[i] / sqrt(mean(x²) + eps) * w[i]`, `eps = 1e-5`.

## RoPE
For each head, pairs `(2i, 2i+1)` rotated by `m * theta^(-2i/head_dim)`,
`m = pos`, `theta = 10000` (configurable via `rope_theta`).

## Attention
- Scale `1/sqrt(head_dim)`, causal (only `0..pos` visible).
- GQA: `kv_head = head / (n_heads / n_kv_heads)`.
- Softmax in FP32, then `out = Σ p * v`.

## SwiGLU FFN
`FFN(h) = W2( SiLU(W1 h) ⊙ (W3 h) )`, no bias anywhere.

## KV cache
`k_cache, v_cache: [n_layers][seq_len][n_kv_heads * head_dim]` FP32.
Pre-allocated zero; written at index `pos` each step.

## Numerical checks
- Baseline vs plugin logits must agree within 1e-4 (FP32) on first 10 tokens.
- `tests/` compares against Karpathy `llama2.c` sampler outputs for stories15M.
