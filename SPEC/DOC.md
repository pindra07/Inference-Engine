# DOC — Download Llama-2 from Hugging Face & Run on Mac

This is the **how-to-run** doc. Theory lives in `01–09`
(`08` = optimization notes, `09` = folder design).

## 0. What you need
- Mac with M1/M2/M3/M4 (Apple Silicon), macOS 13+, Xcode CLT (`xcode-select --install`), CMake ≥ 3.16, Python ≥ 3.9.
- ~15 GB free for 7B (FP32 export is ~26 GB — prefer the small test model first).
- Hugging Face account + access to a Llama-2 repo (Meta gated).

## 1. Build the libraries + Mac plugin

```bash
git clone <this-repo> inference && cd inference
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j   # libinference_common + libinference_llama2 + plugins + run_llama
ls build/models/llama2/plugins/   # -> libplugin_llama2_mac_mseries.dylib, ...
ls build/models/llama2/examples/run_llama
```

(If `cmake` is missing: `pip install cmake`.)

To force the Mac plugin, pass
`--backend build/models/llama2/plugins/libplugin_llama2_mac_mseries.dylib`
(or set `INFERENCE_BACKEND=...`). Without it, the built-in reference backend
is used (slower, same numerics).

## 2. Download Llama-2 from Hugging Face

Checkpoints live in `models/llama2/weights/` (gitignored). Full notes there.

### Option A — small test first (recommended, no approval needed)
Karpathy's TinyStories 15M checkpoint, same architecture, runs on any M1 in seconds.
It uses the standard 32k Llama tokenizer (NOT tok512 — that one is only for stories260K):

```bash
curl -L -o models/llama2/weights/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/llama2/weights/tokenizer.bin \
  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

### Option B — real Llama-2-7B (gated)
1. Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf and accept the license.
2. Create a token at https://huggingface.co/settings/tokens (read-only is enough).
3. Download + convert to our `.bin` format:

```bash
pip install torch safetensors transformers huggingface_hub numpy
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /tmp/Llama-2-7b-hf

# Convert HF safetensors -> FP32 .bin + tokenizer.bin (see tools/export_hf_to_bin.py)
python tools/export_hf_to_bin.py \
  --hf-dir /tmp/Llama-2-7b-hf \
  --out-bin models/llama2/weights/llama2-7b.bin \
  --out-tok models/llama2/weights/tokenizer.bin
ls -lh models/llama2/weights/   # llama2-7b.bin (~26 GB FP32), tokenizer.bin
```

> **8 GB RAM Mac?** FP32 7B will not fit. Either use a 16 GB+ Mac, export a
> smaller model (TinyLlama-1.1B), or wait for INT8 quant (`recommendation.md` rec #1).

## 3. Run on Mac (M1–M4)

```bash
B=build/models/llama2; W=models/llama2/weights
# TinyStories sanity check (fast, ~850 tok/s on M4)
./$B/examples/run_llama \
  --checkpoint $W/stories15M.bin \
  --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --prompt "Once upon a time" --steps 200 --temperature 0.0

# Llama-2-7B chat
./$B/examples/run_llama \
  --checkpoint $W/llama2-7b.bin \
  --tokenizer $W/tokenizer.bin \
  --backend $B/plugins/libplugin_llama2_mac_mseries.dylib \
  --chat --steps 256 --temperature 0.7 --top_p 0.9 \
  --prompt "Explain recursion in one sentence."
```

Expected: first tokens stream immediately; ~2–4 tok/s on M-series for 7B-FP32
(bandwidth wall — see `SPEC/08`), stories15M in the hundreds/tens per backend
(reference ~195, intel ~337, mac ~850 on M4).

### Verification note (correctness gate)
Greedy (`--temperature 0.0`) output on stories15M is **byte-identical** to
`llama2.c`'s `run` on the same prompt across all backends. Two intentional
CLI differences when comparing: our CLI does not echo the prompt, and
`--steps N` counts *generated* tokens (llama2.c `-n` counts prefill+decode).

## 4. CLI flags (`run_llama --help`)
`--checkpoint PATH --tokenizer PATH --backend PATH --prompt STR --chat
 --steps N --temperature F --top_p F --top_k N --seed N --seq_len N`

## 5. Troubleshooting
| Symptom | Fix |
|---|---|
| `dlopen ... image not found` | Use absolute `--backend` path; check `file *.dylib` shows `arm64`. |
| `checkpoint truncated / unexpected trailing size` | You passed an HF file directly — run the export script first. |
| OOM / killed on 8 GB Mac | Use stories15M; 7B-FP32 needs ≥ 32 GB RAM ideally. |
| Slow on M-series | Confirm log shows `backend=llama2/mac_mseries`; else fallback was used. |
| Windows Intel plugin on Mac | Fine for correctness tests (scalar fallback), but use `mac_mseries` for speed. |

## 6. Next steps
- Add your own plugin: copy `models/llama2/plugins/mac_mseries/` →
  `models/llama2/plugins/my_npu/`, subclass `ReferenceBackend`, register via
  `llama2_plugin(...)` (see `SPEC/04_PLUGIN_API.md`).
- Add a new model family: follow the checklist in `SPEC/09_MODEL_LAYOUT.md`.
- Read `SPEC/08_OPTIMIZATION.md` for what was tuned and `recommendation.md`
  for what to do next.
