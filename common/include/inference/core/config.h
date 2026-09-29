#pragma once
// Model hyper-parameters. Mirrors Meta params.json + checkpoint header.

namespace inference {
namespace core {

struct ModelConfig {
  int dim = 0;            // transformer dimension
  int hidden_dim = 0;     // FFN inner dimension
  int n_layers = 0;       // transformer blocks
  int n_heads = 0;        // query heads
  int n_kv_heads = 0;     // key/value heads (GQA); == n_heads for MHA
  int vocab_size = 0;     // vocabulary
  int seq_len = 0;        // max context length
  float rope_theta = 10000.0f;
  bool shared_classifier = false;  // true => reuse embedding as output proj

  int head_dim() const { return dim / n_heads; }
  int kv_dim() const { return (dim * n_kv_heads) / n_heads; }
  bool valid() const {
    return dim > 0 && hidden_dim > 0 && n_layers > 0 && n_heads > 0 &&
           n_kv_heads > 0 && vocab_size > 0 && seq_len > 0 &&
           dim % n_heads == 0;
  }
};

}  // namespace core
}  // namespace inference
