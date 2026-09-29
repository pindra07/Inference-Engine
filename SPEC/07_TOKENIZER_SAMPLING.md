# 07 — Tokenizer & Sampling (shared, `common/`)

## Tokenizer (`common/.../tokenizer/tokenizer.h`)
File format: Karpathy `tokenizer.bin` — `int32 max_token_len`, then per
token: `float32 score`, `int32 len`, `bytes[len]`. Pieces store `' '` where
sentencepiece has `▁` (converters do the replacement); byte pieces look like
`<0x00>`..`<0xFF>`.

`encode()` is llama2.c-compatible:
1. `[BOS]` (unless `add_bos=false`).
2. Optional dummy-prefix `" "` lookup (Llama convention, **on by default**;
   Gemma-family engines turn it off via `set_dummy_prefix(false)` — Gemma
   pieces carry their own spaces, SPEC/10).
3. Registered control pieces first (`add_special`, e.g. `<start_of_turn>`),
   longest match wins — chat markers survive as single ids.
4. UTF-8 codepoint segmentation with byte fallback: known codepoint → its
   id, else each byte → `byte + byte_base`.
5. Best-first BPE merges until no mergeable pair remains. Scores are merge
   priorities: converters write `-rank` from the BPE merge list (true BPE
   order); Llama files carry sentencepiece scores, which order identically
   in practice. Both verified token-for-token against reference tokenizers.

Model-specific adaptations (no format change):
- `byte_base`: Llama packs byte pieces at 3..258; others anywhere. `load()`
  scans for `<0x00>` and uses its index (falls back to 3).
- `set_special_ids(bos, eos)`: Llama default (1, 2); Gemma sets (2, 1).
  `decode(prev, id)` resolves `<0xXX>` to raw bytes and strips the one
  leading space after BOS (sentencepiece rule, mirrors llama2.c).

## Sampler (`common/.../sampling/sampler.h`)
```cpp
struct SampleConfig { float temperature = 1.0f; float top_p = 0.9f;
                      int top_k = 0; uint64_t seed = 42; };
int sample(const float* logits, int vocab, const SampleConfig&, uint64_t* rng_state, float* scratch);
```
- `temperature <= 0` → greedy argmax (also NaN-guarded).
- Else: scale → softmax → top-k (`nth_element` partition + sort the window,
  SPEC/08) → nucleus top-p truncation → multinomial via `xoroshiro128+`
  (deterministic per seed).
- No allocations in the sample path (caller scratch = vocab floats).

## Prompt formats
- Llama-2-Chat: `<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]`
- Gemma: `<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n`
  (+ BOS prepended by the encoder in both cases).
- Interactive mode (`--interactive`, shared `app::run_repl`): each typed
  line is one fresh prompt (no cross-turn KV memory in v1); `quit`/`/quit`/
  `exit`/`:q` or Ctrl+D ends the session.
