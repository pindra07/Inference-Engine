# 06 — Memory & Weight Layout

## Checkpoint format (v1 = Karpathy `llama2.bin` compatible)

```
offset  size  field
0       7×4   int32 header: [dim, hidden_dim, n_layers, n_heads,
                             n_kv_heads, vocab_size, seq_len]
...     ...   float32 weights GROUPED BY TENSOR (matches llama2.c
              memory_map_weights — NOT layer-interleaved):
  token_embedding [vocab, dim]
  ALL layers' attn_rms_w      [n_layers, dim]
  ALL layers' Wq              [n_layers, dim, dim]
  ALL layers' Wk              [n_layers, dim, kv_dim]
  ALL layers' Wv              [n_layers, dim, kv_dim]
  ALL layers' Wo              [n_layers, dim, dim]
  ALL layers' ffn_rms_w       [n_layers, dim]
  ALL layers' W1 (gate)       [n_layers, hidden, dim]
  ALL layers' W2 (down)       [n_layers, dim, hidden]
  ALL layers' W3 (up)         [n_layers, hidden, dim]
  final_rms_w [dim]
  [legacy freq_cis_real, freq_cis_imag — skipped, RoPE done on the fly]
  [Wcls [vocab, dim] unless shared with embedding]
```

Layout variants (all accepted — detected from file size, not flags):
- Modern export: no freq tables; negative `vocab_size` signals shared classifier.
- Legacy export (e.g. tinyllamas `stories*.bin`): freq tables stored after
  `final_rms`, `wcls` omitted (shared) even with positive `vocab_size`.
See `src/core/weights.cc` (`Weights::load`) for the detection logic.

`src/engine/engine.cc` memory-maps the file, reads the header, then binds
`const float*` views into the mapped region — **zero copy**.

`ModelConfig` (`include/inference/core/config.h`) mirrors the header plus
`vocab_size`, `seq_len`, `rope_theta`, `shared_classifier`.

## Alignment
- Engine allocates activations with 64-byte alignment (`posix_memalign` /
  `_aligned_malloc`) so AVX2/NEON loads never fault.
- Mapped weights may be unaligned; backends must handle it (use `cblas` /
  unaligned `_mm256_loadu_ps`).

## Memory budget (FP32, 7B, seq 2048)
- Weights ≈ 26 GB → use FP16 export (`--dtype fp16` in export script → stored as
  FP32? No — v1 requires FP32; for M1-8GB use `stories15M` or 7B-Q stub).
  Practical M1 test: `stories15M.bin` (~60 MB) or TinyLlama.
- KV cache = `2 × layers × seq × kv_dim × 4 bytes`
  (7B/2048 ≈ 2×32×2048×4096×4 ≈ 2.1 GB). Reduce `--seq_len` for small Macs.

## Thread-safety
- `Weights` immutable after load → share across engine instances.
- `LlamaEngine` is **not** thread-safe; one instance per thread/stream.
