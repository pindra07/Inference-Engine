// BPE tokenizer, llama2.c-compatible.
// Format: int32 max_token_len, then per token: float32 score, int32 len,
// bytes[len]. Pieces use ' ' for sentencepiece '▁' (see tokenizer.py).

#include "inference/tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace inference {
namespace tokenizer {

void Tokenizer::load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open tokenizer: " + path);
  int32_t max_len = 0;
  if (fread(&max_len, sizeof(max_len), 1, f) != 1) {
    fclose(f);
    throw std::runtime_error("bad tokenizer header: " + path);
  }
  max_token_len_ = max_len;
  vocab_.clear();
  scores_.clear();
  for (;;) {
    float score = 0;
    int32_t len = 0;
    if (fread(&score, sizeof(score), 1, f) != 1) break;  // EOF
    if (fread(&len, sizeof(len), 1, f) != 1) {
      fclose(f);
      throw std::runtime_error("truncated tokenizer: " + path);
    }
    if (len < 0 || len > 4096) {
      fclose(f);
      throw std::runtime_error("bad token len in: " + path);
    }
    std::string tok((size_t)len, '\0');
    if (len > 0 && fread(tok.data(), 1, (size_t)len, f) != (size_t)len) {
      fclose(f);
      throw std::runtime_error("truncated token bytes: " + path);
    }
    vocab_.push_back(tok);
    scores_.push_back(score);
  }
  fclose(f);
  if (vocab_.empty()) throw std::runtime_error("empty vocab: " + path);
  sorted_.reserve(vocab_.size());
  for (int i = 0; i < (int)vocab_.size(); ++i) sorted_.emplace_back(vocab_[i], i);
  std::sort(sorted_.begin(), sorted_.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  // Byte-fallback pieces look like <0x00>..<0xFF>. Llama packs them at ids
  // 3..258; other families (e.g. Gemma) put them elsewhere — detect the base
  // so encode() stays correct without format changes (see SPEC/10).
  byte_base_ = 3;
  int zero = lookup("<0x00>");
  if (zero >= 0) byte_base_ = zero;
}

int Tokenizer::lookup(const std::string& piece) const {
  auto it = std::lower_bound(
      sorted_.begin(), sorted_.end(), piece,
      [](const std::pair<std::string, int>& e, const std::string& v) {
        return e.first < v;
      });
  if (it != sorted_.end() && it->first == piece) return it->second;
  return -1;
}

void Tokenizer::add_special(const std::string& piece) {
  if (piece.empty()) throw std::runtime_error("empty special piece");
  if (lookup(piece) < 0)
    throw std::runtime_error("special piece not in vocab: " + piece);
  specials_.push_back(piece);
  std::sort(specials_.begin(), specials_.end(),
            [](const std::string& a, const std::string& b) {
              return a.size() > b.size();  // longest match wins
            });
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool add_bos) const {
  std::vector<int> ids;
  ids.reserve(text.size() + 3);
  if (add_bos) ids.push_back(bos_id());

  if (!text.empty()) {
    if (dummy_prefix_) {
      // Dummy prefix (sentencepiece add_dummy_prefix=true, Llama-style).
      // Gemma-family tokenizers carry their own spaces: off, skip this.
      int dummy = lookup(" ");
      if (dummy < 0)
        throw std::runtime_error("tokenizer vocab lacks dummy prefix ' '");
      ids.push_back(dummy);
    }

    // Codepoint segmentation with byte fallback (first 3 Llama ids are
    // <unk>,<s>,</s>; other families use the detected byte_base_).
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
      // Registered control pieces win over everything (chat markers).
      bool special = false;
      for (const std::string& sp : specials_) {
        if (sp.size() <= n - i &&
            std::memcmp(text.data() + i, sp.data(), sp.size()) == 0) {
          ids.push_back(lookup(sp));  // >= 0, guaranteed by add_special
          i += sp.size();
          special = true;
          break;
        }
      }
      if (special) continue;
      // Length of the UTF-8 codepoint starting at i.
      unsigned char c = (unsigned char)text[i];
      size_t cp_len = 1;
      if ((c & 0x80) == 0) {
        cp_len = 1;
      } else if ((c & 0xE0) == 0xC0) {
        cp_len = 2;
      } else if ((c & 0xF0) == 0xE0) {
        cp_len = 3;
      } else if ((c & 0xF8) == 0xF0) {
        cp_len = 4;
      }
      if (i + cp_len > n) cp_len = n - i;  // stray trailing bytes
      std::string piece = text.substr(i, cp_len);
      int id = lookup(piece);
      if (id >= 0) {
        ids.push_back(id);
      } else {
        for (size_t k = 0; k < cp_len; ++k)
          ids.push_back((unsigned char)text[i + k] + byte_base_);
      }
      i += cp_len;
    }

    // Score-ordered BPE merges (best pair first, repeat until none).
    std::string buf;
    buf.reserve((size_t)max_token_len_ * 2 + 4);
    while (ids.size() >= 2) {
      float best_score = -1e10f;
      int best_id = -1;
      size_t best_idx = 0;
      for (size_t j = 0; j + 1 < ids.size(); ++j) {
        buf.assign(vocab_[(size_t)ids[j]]);
        buf.append(vocab_[(size_t)ids[j + 1]]);
        int id = lookup(buf);
        if (id >= 0 && scores_[(size_t)id] > best_score) {
          best_score = scores_[(size_t)id];
          best_id = id;
          best_idx = j;
        }
      }
      if (best_id < 0) break;
      ids[best_idx] = best_id;
      ids.erase(ids.begin() + (long)best_idx + 1);
    }
  }
  return ids;
}

std::string Tokenizer::decode(int prev_token, int token) const {
  if (token < 0 || token >= (int)vocab_.size()) return "";
  const std::string& piece = vocab_[(size_t)token];
  // Raw byte tokens look like <0x01>.
  unsigned byte_val = 0;
  if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
      piece[2] == 'x' && piece[5] == '>' &&
      sscanf(piece.c_str() + 3, "%2x", &byte_val) == 1) {
    return std::string(1, (char)byte_val);
  }
  // Sentencepiece strips one leading space right after BOS.
  if (prev_token == bos_id() && !piece.empty() && piece[0] == ' ')
    return piece.substr(1);
  return piece;
}

}  // namespace tokenizer
}  // namespace inference
