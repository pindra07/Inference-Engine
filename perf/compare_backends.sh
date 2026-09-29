#!/usr/bin/env bash
# Compare all backends for one model and print a summary table.
#   ./perf/compare_backends.sh llama2 ["prompt"] [steps]
#   ./perf/compare_backends.sh gemma2 ["prompt"] [steps]
# Appends per-repeat CSV to perf/results/<model>-<timestamp>.csv.
# macOS bash 3.2 compatible (no associative arrays).
set -u

MODEL="${1:?usage: compare_backends.sh llama2|gemma2 [prompt] [steps]}"
PROMPT="${2:-Once upon a time}"
STEPS="${3:-128}"

# Default checkpoints (override with CKPT=... TOK=... env vars).
case "$MODEL" in
  llama2)
    CKPT="${CKPT:-models/llama2/weights/stories15M.bin}"
    TOK="${TOK:-models/llama2/weights/tokenizer.bin}" ;;
  gemma2)
    CKPT="${CKPT:-models/gemma2/weights/tiny-gemma2.bin}"
    TOK="${TOK:-models/gemma2/weights/tokenizer.bin}" ;;
  *)
    echo "unknown model: $MODEL (want llama2|gemma2)" >&2
    exit 1 ;;
esac
if [ ! -f "$CKPT" ] || [ ! -f "$TOK" ]; then
  echo "missing checkpoint/tokenizer: $CKPT $TOK" >&2
  echo "(override with CKPT=... TOK=...)" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin) EXT=dylib ;;
  MINGW*|MSYS*|CYGWIN*) EXT=dll ;;
  *) EXT=so ;;
esac

BENCH="build/perf/bench_${MODEL}"
# Binary names drop the trailing "2" (bench_llama, bench_gemma).
case "$MODEL" in
  llama2) BENCH="build/perf/bench_llama" ;;
  gemma2) BENCH="build/perf/bench_gemma" ;;
esac
PLUGDIR="build/models/${MODEL}/plugins"
if [ ! -x "$BENCH" ]; then
  echo "missing $BENCH — build first: cmake --build build -j" >&2
  exit 1
fi

# Pairs of "label|plugin-path" (empty path = built-in backend).
BACKENDS="reference|
cpu_baseline|${PLUGDIR}/libplugin_${MODEL}_cpu_baseline.${EXT}
mac_mseries|${PLUGDIR}/libplugin_${MODEL}_mac_mseries.${EXT}
windows_intel|${PLUGDIR}/libplugin_${MODEL}_windows_intel.${EXT}"

OUT="perf/results/${MODEL}-$(date +%Y%m%d-%H%M%S).csv"
mkdir -p perf/results
FIRST=1
SUMMARY="$(mktemp)"
trap 'rm -f "$SUMMARY"' EXIT

echo "model=$MODEL prompt=\"$PROMPT\" steps=$STEPS -> $OUT"
while IFS='|' read -r label path; do
  [ -z "$label" ] && continue
  if [ -n "$path" ] && [ ! -f "$path" ]; then
    echo "skip $label (missing $path)" >&2
    continue
  fi
  # shellcheck disable=SC2086
  if [ -n "$path" ]; then
    RUN_OUT="$("$BENCH" --checkpoint "$CKPT" --tokenizer "$TOK" \
      --backend "$path" --prompt "$PROMPT" --steps "$STEPS" --format csv)"
  else
    RUN_OUT="$("$BENCH" --checkpoint "$CKPT" --tokenizer "$TOK" \
      --prompt "$PROMPT" --steps "$STEPS" --format csv)"
  fi
  if [ "$FIRST" -eq 1 ]; then
    echo "$RUN_OUT" | head -1 > "$OUT"
    FIRST=0
  fi
  echo "$RUN_OUT" | tail -n +2 >> "$OUT"
  # best decode_tps + mean ttft for the summary table
  echo "$RUN_OUT" | tail -n +2 | awk -F, -v label="$label" '
    { tps[NR]=$8; ttft+=$7; n=NR }
    END {
      best=tps[1]; for (i in tps) if (tps[i]>best) best=tps[i];
      printf "%-14s best %8.1f tok/s | ttft mean %7.1f ms\n", label, best, ttft/n*1000;
    }' | tee -a "$SUMMARY"
done <<EOF
$BACKENDS
EOF

echo "---- summary (best of repeats) ----"
cat "$SUMMARY"
echo "full per-repeat data: $OUT"
