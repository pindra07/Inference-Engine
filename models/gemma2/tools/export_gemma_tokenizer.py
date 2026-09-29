#!/usr/bin/env python3
"""Export a Gemma SentencePiece model -> tokenizer.bin (Karpathy layout).

BPE correctness (SPEC/10): sentencepiece BPE merges by learned RANK, not by
piece score, so scores here are synthesized as -rank from tokenizer.json's
merge list (earlier merge = higher score = merged first by our best-first
loop). Same ▁→' ' replacement as Karpathy's tokenizer.py, applied to pieces
AND merge sides. HF added_tokens (control markers) are appended at exact ids.

Usage:
  python models/gemma2/tools/export_gemma_tokenizer.py \
      --model tokenizer.model --json tokenizer.json \
      --out models/gemma2/weights/tokenizer.bin
Requires: sentencepiece.
"""
import argparse
import json
import struct

REPLACEMENT = "\u2581"
UNUSED_SCORE = -1e30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="sentencepiece .model")
    ap.add_argument("--json", default=None, help="HF tokenizer.json (added tokens)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=args.model)
    pieces = [sp.id_to_piece(i).replace(REPLACEMENT, " ")
              for i in range(sp.vocab_size())]
    index_of = {p: i for i, p in enumerate(pieces)}

    # Merge-rank scores: earlier merge wins. Missing concats are warned
    # (a consistent BPE model shouldn't have any).
    scores = [UNUSED_SCORE] * len(pieces)
    tj = json.load(open(args.json)) if args.json else None
    if tj is not None:
        missing = 0
        for rank, (a, b) in enumerate(tj["model"]["merges"]):
            concat = (a.replace(REPLACEMENT, " ") +
                      b.replace(REPLACEMENT, " "))
            i = index_of.get(concat)
            if i is None:
                missing += 1
                continue
            if scores[i] == UNUSED_SCORE:
                scores[i] = -float(rank)
        if missing:
            print(f"warning: {missing} merges have no target piece")

    if tj is not None:
        added = sorted(tj.get("added_tokens", []), key=lambda a: a["id"])
        n_new = 0
        for a in added:
            content = a["content"].replace(REPLACEMENT, " ")
            if a["id"] < len(pieces):
                # Already in the base vocab (e.g. <pad>/<eos>/<bos>).
                if pieces[a["id"]] != content:
                    raise SystemExit(
                        f"added token {a['content']} id {a['id']} conflicts "
                        f"with base piece {pieces[a['id']]!r}")
                continue
            if a["id"] != len(pieces):
                raise SystemExit(
                    f"added token {a['content']} id {a['id']} not contiguous "
                    f"(expected {len(pieces)})")
            pieces.append(content)
            scores.append(0.0)
            n_new += 1
        print(f"+{n_new} new added tokens "
              f"(skipped {len(added)-n_new} already present)")

    max_len = max(len(p.encode("utf8")) for p in pieces)
    with open(args.out, "wb") as f:
        f.write(struct.pack("i", max_len))
        for s, p in zip(scores, pieces):
            b = p.encode("utf8")
            f.write(struct.pack("f", float(s)))
            f.write(struct.pack("i", len(b)))
            f.write(b)
    bb = pieces.index("<0x00>") if "<0x00>" in pieces else None
    print(f"wrote {args.out}: {len(pieces)} pieces, byte_base=<0x00>@{bb}")


if __name__ == "__main__":
    main()
