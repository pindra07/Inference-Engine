# 10 — Gemma2 Model Support

Reference: HF `transformers` `modeling_gemma2.py` / `configuration_gemma2.py`
(read verbatim during design — discrepancies here are bugs, fix the code).

## Architecture deltas vs Llama-2 (all verified against HF source)

| # | Llama-2 | Gemma2 | Code impact |
|---|---|---|---|
| 1 | RMSNorm `x·w`, eps 1e-5 | `Gemma2RMSNorm`: `x·rsqrt(mean(x²)+eps)·(1+w)`, eps **1e-6** | Folded `(1+w)` at load into owned buffers → same `rmsnorm` kernel, ABI untouched (`weights.cc`) |
| 2 | 1 norm + residual per block | **4 norms**: `input → attn → post_attn →(+res) → pre_ffn → mlp → post_ffn →(+res)` | `transformer.cc` graph |
| 3 | SwiGLU (`silu`) | **GeGLU** with `gelu_pytorch_tanh`: `down(gelu(gate)·up)` | New ABI kernel `gelu_tanh` (exact HF formula, incl. 0.044715 const) |
| 4 | MHA/GQA full causal | **Interleaved sliding/full**: layer `l` sliding iff `(l+1)%2==1`, window 4096 | `attention_softcap(..., window)` masks to `[pos-W+1..pos]`; full cache kept (v1 simplification, same numerics) |
| 5 | attn scale `1/√head_dim` | scale = **`query_pre_attn_scalar^-0.5`** (e.g. 1/16, 1/12 — NOT head_dim) | Engine passes `attn_scale` explicitly; never derived |
| 6 | No logit capping | **Softcapping**: attn logits `cap·tanh(x/cap)` (50), final logits (30); `≤0` = off | New ABI kernels `attention_softcap`, `tanh_softcap` |
| 7 | `head_dim = dim/heads`, q ≡ dim | **Explicit `head_dim`** (256); q=`H·hd` may ≠ dim (9B: 4096 vs 3584) | `RunState.q` sized `[q_dim]`; `Wo` is `[dim × q_dim]`; KV `[kv_dim]` |
| 8 | No embedding scale | `emb[token] · √dim` (`Gemma2TextScaledWordEmbedding`) | `transformer.cc` one multiply after `memcpy` |
| 9 | No QK-norm | **No QK-norm** (that's Gemma3) — confirmed absent in HF source | Nothing to do |
| 10 | Biases: none | `attention_bias=false` assumed | Loader/export assert no-bias; biases are a v2 ABI item |
| 11 | Tied-or-not via file size | **`tie_word_embeddings=true`** (default); untied supported via trailing `wcls` like llama2 | Loader detects from file size |

RoPE formula itself is identical to Llama-2 (same `rope` kernel, `rope_theta`
from file, default 10000).

## Checkpoint format (ours): `GGM2` v1

New container (not Karpathy-compatible — Gemma needs more header fields).
All ints little-endian int32, floats FP32:

```
offset  contents
0       magic "GGM2" (4 bytes) + int32 version (=1)
8       int32[9]: dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                  head_dim, vocab_size, seq_len, sliding_window
44      float32[5]: rope_theta, rms_eps, attn_scale (=query_pre_attn_scalar^-0.5),
                    attn_softcap, final_softcap
64      weights, GROUPED BY TENSOR (same doctrine as llama2, SPEC/06):
        tok_emb [vocab,dim]
        ALL input_rms, ALL Wq [L,dim,q_dim], ALL Wk, ALL Wv, ALL Wo [L,q_dim,dim],
        ALL post_attn_rms, ALL pre_ffn_rms,
        ALL Wgate [L,hidden,dim], ALL Wdown [L,dim,hidden], ALL Wup,
        ALL post_ffn_rms, final_rms [dim]
        [wcls [vocab,dim] iff file has trailing bytes for it]
```

`models/gemma2/tools/export_hf_to_gemma2.py` converts any HF Gemma2
`safetensors` dir to this (plus asserts no-bias, reads theta/eps/caps from
`config.json`). Norm `(1+w)` folding happens in the **loader**, not the
exporter, so the file stays a faithful dump of HF tensors.

## Tokenizer notes (Gemma pitfalls)

- Vocab 256000 (SentencePiece BPE, same algorithm as Llama → same encoder).
- **Byte fallback is NOT at +3.** Loader scans vocab for the `<0x00>` piece
  and uses its index as `byte_base` (llama files have no such piece → 3).
- **Special ids differ**: bos=2, eos=1 (llama: 1,2). `Tokenizer` keeps
  llama defaults; gemma engine calls `set_special_ids(2, 1)` after load.
- **Dummy prefix**: converter replaces `▁`→`' '` in pieces (same as
  Karpathy's `tokenizer.py`), so the `" "` lookup succeeds.
- **Control tokens** (`<start_of_turn>` etc.): converter appends HF
  `added_tokens` at their exact ids (positional format requires index==id).
- Chat template (`--chat`): `<start_of_turn>user\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n`
  with BOS prepended by the encoder (matches HF `add_bos_token` behavior).

## ABI v1.1 additions (see SPEC/04)

`attention_softcap(..., softcap, window)` and `tanh_softcap(x, n, cap)` and
`gelu_tanh(x, n)`. Implemented scalar in `ReferenceBackend`; mac overrides
the hot one (`attention_softcap`); intel inherits attention (matvec still
accelerated = 95% of runtime). v1's 7 methods are untouched — llama2
binaries and verification stand as-is.

## Verification (this pass — all green)

1. **Gold standard PASSED**: `yujiepan/gemma-2-tiny-random` (hidden=8,
   2 layers, sliding+full, GQA 8/4, caps 50/30) → converted → greedy TOP5
   **bit-exact vs independent pure-torch reference** (FP32, same prompt:
   `43377:1.2439 235958:1.2021 ...` identical), and 6-token greedy decode
   text identical (`' rugbyjsk書の書の書の書の'`). Covers: folded norms,
   GeGLU, softcaps, sliding mask, explicit head_dim with q≠dim, GQA,
   rope de-permutation, embedding scale, tied logits, 256k tokenizer.
2. **All 4 backends agree**: builtin reference + cpu/mac/intel plugins
   produce identical greedy text (validates mac `attention_softcap`,
   `tanh_softcap`/vForce and intel AVX2 paths bit-exact).
3. **Tokenizer parity**: our encode matches HF `tokenizers` output
   id-for-id on multi-word prompts ("Hello world", "Say hello to my
   friend", "The quick brown fox").
4. **Llama untouched**: stories15M greedy story still byte-identical to
   `llama2.c` after all shared-code changes (ABI is purely additive).
5. Real `gemma-2-2b/9b` bit-check left to a machine with HF gated access;
   `models/gemma2/weights/README.md` gives exact commands.

## Lessons (bugs found by differential testing)

- **Bind views from valid dims.** `Weights::bind_folded_views()` first read
  `config_.base.n_layers`, which is still 0 when `load()` binds — silently
  aliasing every norm view to layer 0. It now takes `n_layers` explicitly
  plus a fail-fast size check. Symptom pattern to remember: norm-free
  quantities (attention out, Wo out, embeddings) matched while everything
  downstream of a norm diverged.
- **BPE scores ≠ merge order.** Sentencepiece BPE encodes by merge *rank*;
  raw piece scores mis-merge Gemma vocabs (llama happened to work). The
  converter now writes `-rank` from `tokenizer.json`'s merge list.
- **Gemma has no dummy prefix.** Sentence-initial words are bare pieces;
  prepending `" "` (llama2.c behavior) corrupts the first word. Engine flag
  `set_dummy_prefix(false)`; default stays llama-compatible.
