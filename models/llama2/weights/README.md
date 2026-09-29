# Llama-2 weights — put checkpoints + tokenizer here (gitignored).
# This folder intentionally holds no source; see SPEC/DOC.md.

## TinyStories 15M (no approval needed, runs anywhere)
```bash
curl -L -o models/llama2/weights/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -L -o models/llama2/weights/tokenizer.bin \
  https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```
NOTE: stories15M uses the 32k `tokenizer.bin`, NOT `tok512.bin`
(that one is only for stories260K).

## Real Llama-2-7B (gated)
Request access at https://huggingface.co/meta-llama/Llama-2-7b-hf, then:
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir /tmp/Llama-2-7b-hf
python tools/export_hf_to_bin.py --hf-dir /tmp/Llama-2-7b-hf \
  --out-bin models/llama2/weights/llama2-7b.bin \
  --out-tok models/llama2/weights/tokenizer.bin
```
