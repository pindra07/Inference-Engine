# Gemma2 weights — put checkpoints + tokenizer here (gitignored).

## Tiny random test model (no approval needed, ~10 MB)
Ungated HF repo with a 2-layer Gemma2 config — for plumbing + math
verification against `transformers` (see SPEC/10):

```bash
huggingface-cli download yujiepan/gemma-2-tiny-random \
  --local-dir /tmp/gemma-2-tiny-random
python models/gemma2/tools/export_hf_to_gemma2.py \
  --hf-dir /tmp/gemma-2-tiny-random \
  --out-bin models/gemma2/weights/tiny-gemma2.bin
python models/gemma2/tools/export_gemma_tokenizer.py \
  --model /tmp/gemma-2-tiny-random/tokenizer.model \
  --json /tmp/gemma-2-tiny-random/tokenizer.json \
  --out models/gemma2/weights/tokenizer.bin
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/tiny-gemma2.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --prompt "Hello" --steps 20 --temperature 0.0
```

## Real Gemma-2-2B / 9B (gated, free)
1. Accept the license at https://huggingface.co/google/gemma-2-2b (or
   `google/gemma-2-9b`) and create a read token.
2. Download + convert (same scripts, real weights):
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
huggingface-cli download google/gemma-2-2b --local-dir /tmp/gemma-2-2b
python models/gemma2/tools/export_hf_to_gemma2.py \
  --hf-dir /tmp/gemma-2-2b \
  --out-bin models/gemma2/weights/gemma2-2b.bin
python models/gemma2/tools/export_gemma_tokenizer.py \
  --model /tmp/gemma-2-2b/tokenizer.model \
  --json /tmp/gemma-2-2b/tokenizer.json \
  --out models/gemma2/weights/tokenizer.bin
./build/models/gemma2/examples/run_gemma \
  --checkpoint models/gemma2/weights/gemma2-2b.bin \
  --tokenizer models/gemma2/weights/tokenizer.bin \
  --backend build/models/gemma2/plugins/libplugin_gemma2_mac_mseries.dylib \
  --chat --interactive --temperature 0.7
```
