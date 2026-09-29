#pragma once
// BPE tokenizer: Karpathy tokenizer.bin format + llama2.c-compatible
// encode (dummy prefix, byte fallback, score-ordered merges).
// See SPEC/07_TOKENIZER_SAMPLING.md.

#include <string>
#include <utility>
#include <vector>

namespace inference {
namespace tokenizer {

class Tokenizer {
 public:
  Tokenizer() = default;
  void load(const std::string& path);  // throws on error

  // Special ids default to Llama's (bos=1, eos=2). Gemma-family models call
  // set_special_ids(2, 1) after load (see SPEC/10_GEMMA2.md).
  void set_special_ids(int bos, int eos) {
    bos_ = bos;
    eos_ = eos;
  }
  int bos_id() const { return bos_; }
  int eos_id() const { return eos_; }
  int vocab_size() const { return (int)vocab_.size(); }
  // First id of the <0x00>..<0xFF> byte-fallback run (3 for Llama,
  // auto-detected for others at load).
  int byte_base() const { return byte_base_; }
  // SentencePiece dummy prefix (" " prepended, Llama convention, default
  // on). Gemma-family tokenizers have none — the engine turns it off.
  void set_dummy_prefix(bool on) { dummy_prefix_ = on; }

  // llama2.c-compatible encode: [BOS] + optional dummy-prefix " " +
  // codepoints with byte-fallback, then best-first BPE merges. Scores are
  // merge priorities (converters write -rank for BPE vocabs). Set
  // add_bos=false to skip BOS (merges still apply).
  std::vector<int> encode(const std::string& text, bool add_bos = true) const;

  // Decode one token given the previous token id. Resolves <0xXX> byte
  // tokens to raw bytes and strips the sentencepiece leading space that
  // follows BOS (mirrors llama2.c decode()).
  std::string decode(int prev_token, int token) const;

  // Register a control piece (e.g. "<start_of_turn>") that must already
  // exist in vocab. Registered pieces match before codepoint/byte logic, so
  // chat markers encode as single ids instead of byte-fallback garbage.
  // No-op for models that never call it (llama2).
  void add_special(const std::string& piece);

 private:
  // Binary search over sorted (piece, id) index. Returns id or -1.
  int lookup(const std::string& piece) const;

  std::vector<std::string> vocab_;
  std::vector<float> scores_;
  std::vector<std::pair<std::string, int>> sorted_;  // (piece, id)
  std::vector<std::string> specials_;  // control pieces, longest first
  int max_token_len_ = 0;
  int bos_ = 1;
  int eos_ = 2;
  int byte_base_ = 3;
  bool dummy_prefix_ = true;
};

}  // namespace tokenizer
}  // namespace inference
