#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <string>
#include <thread>
#include <atomic>
#include <cstring>
#include <set>

using namespace Rcpp;

// Token structure for storing vocab index and original position
struct Token {
  int vocab_idx;      // Index in vocabulary (0-indexed)
  int original_pos;   // Position in original document before filtering
};

// Constants for sigmoid lookup table
constexpr int EXP_TABLE_SIZE = 1000;
constexpr float MAX_EXP = 6.0f;

// Build exp/sigmoid table like original word2vec
// Stores f(x) = sigmoid(x) = exp(x) / (exp(x) + 1)
inline void build_exp_table(std::vector<float>& exp_table) {
  exp_table.resize(EXP_TABLE_SIZE);
  for (int i = 0; i < EXP_TABLE_SIZE; ++i) {
    float x = (static_cast<float>(i) / EXP_TABLE_SIZE * 2.0f - 1.0f) * MAX_EXP;
    float exp_val = std::exp(x);
    exp_table[i] = exp_val / (exp_val + 1.0f);  // sigmoid
  }
}

// Fast sigmoid lookup
inline float fast_sigmoid(float x, const std::vector<float>& exp_table) {
  if (x < -MAX_EXP) return 0.0f;
  if (x > MAX_EXP) return 1.0f;
  int idx = static_cast<int>((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0f));
  return exp_table[idx];
}

// Build vocabulary from tokens
// NOTE: Vocabulary filtering is done in R - this function just builds indices
void build_vocab(
    const List& tokens_list,
    const IntegerVector& vocab,  // 1-indexed quanteda type indices to include
    std::vector<int>& type_to_vocab,  // Maps quanteda type index to our vocab index
    std::vector<int>& vocab_indices_out,  // Output: ordered vocab indices (1-indexed)
    std::vector<long long>& word_counts_out,
    long long& total_words
) {
  // Get number of types from tokens object
  int n_types = 0;
  for (int doc = 0; doc < tokens_list.size(); ++doc) {
    SEXP doc_sexp = tokens_list[doc];
    if (TYPEOF(doc_sexp) != INTSXP) continue;
    IntegerVector doc_tokens = as<IntegerVector>(doc_sexp);
    for (int i = 0; i < doc_tokens.size(); ++i) {
      int type_idx = doc_tokens[i] - 1;
      if (type_idx >= n_types) n_types = type_idx + 1;
    }
  }
  
  // Count word frequencies
  std::vector<long long> type_counts(n_types, 0);
  total_words = 0;
  
  for (int doc = 0; doc < tokens_list.size(); ++doc) {
    SEXP doc_sexp = tokens_list[doc];
    if (TYPEOF(doc_sexp) != INTSXP) continue;
    
    IntegerVector doc_tokens = as<IntegerVector>(doc_sexp);
    for (int i = 0; i < doc_tokens.size(); ++i) {
      int type_idx = doc_tokens[i] - 1;  // quanteda uses 1-indexing
      if (type_idx >= 0 && type_idx < n_types) {
        type_counts[type_idx]++;
        total_words++;
      }
    }
  }
  
  // Build set of allowed type indices (vocab contains 1-indexed quanteda type indices)
  std::set<int> allowed_types;
  for (int i = 0; i < vocab.size(); ++i) {
    allowed_types.insert(vocab[i] - 1);  // Convert to 0-indexed
  }
  
  // Build vocabulary in the order provided by R (already filtered and ordered)
  // No need to sort - R has already done the filtering with vocab_size/coverage/min_count
  int vocab_len = vocab.size();
  word_counts_out.resize(vocab_len);
  vocab_indices_out.resize(vocab_len);
  type_to_vocab.assign(n_types, -1);  // -1 means not in vocab
  
  for (int vocab_idx = 0; vocab_idx < vocab_len; ++vocab_idx) {
    int type_idx = vocab[vocab_idx] - 1;  // Convert from 1-indexed to 0-indexed
    
    if (type_idx >= 0 && type_idx < n_types) {
      word_counts_out[vocab_idx] = type_counts[type_idx];
      type_to_vocab[type_idx] = vocab_idx;
      vocab_indices_out[vocab_idx] = vocab[vocab_idx];  // Keep as 1-indexed
    }
  }
}

// Calculate weight based on distance
inline float calculate_weight(float dist, float window, const std::string& type, 
                               float param, const std::vector<float>& weights_vec, 
                               int weights_mode, bool include_target, int offset) {
  // Custom weight vector
  if (weights_mode > 0) {
    int idx = -1;
    if (weights_mode == 1) {  // 1..W
      idx = std::abs(offset) - 1;
      if (idx >= 0 && idx < static_cast<int>(weights_vec.size())) return weights_vec[idx];
    } else if (weights_mode == 2) {  // 0..W
      idx = std::abs(offset);
      if (idx >= 0 && idx < static_cast<int>(weights_vec.size())) return weights_vec[idx];
    } else if (weights_mode == 3) {  // -W..-1, 1..W
      int win = (weights_vec.size()) / 2;
      idx = (offset < 0) ? (offset + win) : (win + offset - 1);
      if (idx >= 0 && idx < static_cast<int>(weights_vec.size())) return weights_vec[idx];
    } else if (weights_mode == 4) {  // -W..W
      int win = weights_vec.size() / 2;
      idx = offset + win;
      if (idx >= 0 && idx < static_cast<int>(weights_vec.size())) return weights_vec[idx];
    }
    return 0.0f;
  }
  
  // Decay functions
  if (dist > window) return 0.0f;
  
  if (type == "linear") {
    return std::max(0.0f, (window - dist + 1.0f) / window);
  } else if (type == "harmonic") {
    return (dist == 0.0f) ? 1.0f : 1.0f / dist;
  } else if (type == "exponential") {
    return std::exp(-param * dist);
  } else if (type == "power") {
    return (dist == 0.0f) ? 1.0f : std::pow(dist, -param);
  } else {  // none
    return 1.0f;
  }
}

// Distance calculation functions for different modes

// Clean distance: use original positions (word count)
inline float calculate_distance_clean_words(int pos1, int pos2, 
                                            const std::vector<Token>& sentence) {
  return static_cast<float>(std::abs(sentence[pos2].original_pos - sentence[pos1].original_pos));
}

// Dirty distance: count retained tokens only
inline float calculate_distance_dirty_words(int pos1, int pos2) {
  return static_cast<float>(std::abs(pos2 - pos1));
}

// Clean distance with type widths: sum widths in original document
inline float calculate_distance_clean_with_widths(int pos1, int pos2,
                                                   const std::vector<Token>& sentence,
                                                   const IntegerVector& original_tokens,
                                                   const std::vector<float>& type_widths,
                                                   int doc_start_pos) {
  int orig_pos1 = sentence[pos1].original_pos;
  int orig_pos2 = sentence[pos2].original_pos;
  
  int start = std::min(orig_pos1, orig_pos2);
  int end = std::max(orig_pos1, orig_pos2);
  
  float dist = 0.0f;
  for (int i = start; i < end; ++i) {
    int type_idx = original_tokens[doc_start_pos + i] - 1;  // Convert from 1-indexed
    if (type_idx >= 0 && type_idx < static_cast<int>(type_widths.size())) {
      dist += type_widths[type_idx];
    } else {
      dist += 1.0f;
    }
  }
  return dist;
}

// Dirty distance with type widths: sum widths of retained tokens
inline float calculate_distance_dirty_with_widths(int pos1, int pos2,
                                                   const std::vector<Token>& sentence,
                                                   const std::vector<int>& vocab_to_type,
                                                   const std::vector<float>& type_widths) {
  if (pos1 == pos2) return 0.0f;
  
  int start = std::min(pos1, pos2);
  int end = std::max(pos1, pos2);
  
  float dist = 0.0f;
  for (int i = start; i < end; ++i) {
    if (i == start) {
      dist = 1.0f;
    } else {
      int vocab_idx = sentence[i].vocab_idx;
      if (vocab_idx >= 0 && vocab_idx < static_cast<int>(vocab_to_type.size())) {
        int type_idx = vocab_to_type[vocab_idx];
        if (type_idx >= 0 && type_idx < static_cast<int>(type_widths.size())) {
          dist += type_widths[type_idx];
        } else {
          dist += 1.0f;
        }
      } else {
        dist += 1.0f;
      }
    }
  }
  return dist;
}

// Thread worker for streaming SGNS (word2vec style)
void sgns_streaming_worker(
    // Data (shared, read-only)
    const std::vector<std::vector<Token>>& sentences,  // Tokens with positions
    const std::vector<long long>& word_counts,
    const std::vector<int>& neg_table,
    const int neg_table_size,
    const std::vector<float>& exp_table,
    const std::vector<float>& type_widths,
    const std::vector<int>& vocab_to_type,
    const std::vector<IntegerVector>& original_docs,
    const std::vector<int>& doc_start_positions,
    // Embeddings (shared, read-write)
    float* word_emb,
    float* context_emb,
    // Parameters
    const int n_dims,
    const int n_neg,
    const int window,
    const float initial_lr,
    const long long total_train_words,
    const int thread_id,
    const int n_threads,
    const int start_sent,
    const int end_sent,
    // Weighting parameters
    const std::string& weights_type,
    const float weights_alpha,
    const std::vector<float>& weights_vec,
    const int weights_mode,
    const bool include_target,
    const float forward_weight,
    const float backward_weight,
    const bool use_fast_path,
    const bool use_words_metric,
    const bool use_clean_distance,
    // Progress tracking
    std::atomic<long long>& processed_words
) {
  // Fast RNG (Linear Congruential Generator like word2vec)
  unsigned long long next_random = thread_id + 1;
  
  // Thread-local gradient accumulator
  std::vector<float> hidden_errors(n_dims, 0.0f);
  
  // Thread-local RNG for weighted sampling
  std::mt19937 rng(thread_id + 12345);
  std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
  
  long long local_word_count = 0;
  float alpha = initial_lr;
  
  // Process sentences assigned to this thread
  for (int sent_idx = start_sent; sent_idx < end_sent; ++sent_idx) {
    const std::vector<Token>& sentence = sentences[sent_idx];
    
    // Find doc_start_pos for clean distance with type widths
    int doc_start_pos = -1;
    if (!use_words_metric && use_clean_distance && !original_docs.empty()) {
      // Binary search for the document containing this sentence
      auto it = std::upper_bound(doc_start_positions.begin(), doc_start_positions.end(), sent_idx);
      if (it != doc_start_positions.begin()) {
        --it;
        doc_start_pos = std::distance(doc_start_positions.begin(), it);
      }
    }
    
    // Skip-gram training on this sentence
    for (size_t pos = 0; pos < sentence.size(); ++pos) {
      int word_idx = sentence[pos].vocab_idx;
      
      local_word_count++;
      
      // Update learning rate every 10000 words
      if (local_word_count % 10000 == 0) {
        long long global_word_count = processed_words.load(std::memory_order_relaxed);
        alpha = initial_lr * (1.0f - static_cast<float>(global_word_count) / total_train_words);
        if (alpha < initial_lr * 0.0001f) alpha = initial_lr * 0.0001f;
      }
      
      if (use_fast_path) {
        // Fast path: original word2vec with random window
        next_random = next_random * 25214903917ULL + 11;
        int actual_window = (next_random >> 16) % window;
        
        for (int a = actual_window; a < window * 2 + 1 - actual_window; ++a) {
          if (a == window) continue;
          
          int c_pos = pos - window + a;
          if (c_pos < 0 || c_pos >= static_cast<int>(sentence.size())) continue;
          
          int context_idx = sentence[c_pos].vocab_idx;
          float* w_vec = word_emb + context_idx * n_dims;
          std::memset(hidden_errors.data(), 0, n_dims * sizeof(float));
          
          for (int d = 0; d <= n_neg; ++d) {
            int target;
            float label;
            
            if (d == 0) {
              target = word_idx;
              label = 1.0f;
            } else {
              next_random = next_random * 25214903917ULL + 11;
              target = neg_table[(next_random >> 16) % neg_table_size];
              if (target == word_idx) continue;
              label = 0.0f;
            }
            
            float* c_vec = context_emb + target * n_dims;
            
            float dot = 0.0f;
            for (int k = 0; k < n_dims; ++k) {
              dot += w_vec[k] * c_vec[k];
            }
            
            float pred = fast_sigmoid(dot, exp_table);
            float grad = (label - pred) * alpha;
            
            for (int k = 0; k < n_dims; ++k) {
              hidden_errors[k] += grad * c_vec[k];
            }
            
            for (int k = 0; k < n_dims; ++k) {
              c_vec[k] += grad * w_vec[k];
            }
          }
          
          for (int k = 0; k < n_dims; ++k) {
            w_vec[k] += hidden_errors[k];
          }
        }
      } else {
        // Weighted sampling path
        // Build context sampling weights for this position
        std::vector<float> context_weights;
        std::vector<int> context_positions;
        float total_weight = 0.0f;
        
        for (int offset = -window; offset <= window; ++offset) {
          if (offset == 0 && !include_target) continue;
          
          int c_pos = pos + offset;
          if (c_pos < 0 || c_pos >= static_cast<int>(sentence.size())) continue;
          
          // Calculate distance based on metric and mode
          float dist;
          if (use_words_metric) {
            if (use_clean_distance) {
              dist = calculate_distance_clean_words(pos, c_pos, sentence);
            } else {
              dist = calculate_distance_dirty_words(pos, c_pos);
            }
          } else {
            if (use_clean_distance) {
              if (doc_start_pos >= 0 && doc_start_pos < static_cast<int>(original_docs.size())) {
                dist = calculate_distance_clean_with_widths(pos, c_pos, sentence,
                                                            original_docs[doc_start_pos],
                                                            type_widths, 0);
              } else {
                dist = calculate_distance_clean_words(pos, c_pos, sentence);
              }
            } else {
              dist = calculate_distance_dirty_with_widths(pos, c_pos, sentence,
                                                          vocab_to_type, type_widths);
            }
          }
          
          // Calculate base weight
          float weight = calculate_weight(dist, static_cast<float>(window), weights_type,
                                          weights_alpha, weights_vec, weights_mode, 
                                          include_target, offset);
          
          // Apply directional weighting
          if (offset < 0) weight *= backward_weight;
          if (offset > 0) weight *= forward_weight;
          
          if (weight > 0.0f) {
            context_weights.push_back(weight);
            context_positions.push_back(c_pos);
            total_weight += weight;
          }
        }
        
        if (context_weights.empty()) continue;
        
        // Normalize weights to probabilities
        for (float& w : context_weights) {
          w /= total_weight;
        }
        
        // Sample contexts proportional to weights
        // Number of samples = window (same average as fast path)
        int n_samples = window;
        
        for (int sample = 0; sample < n_samples; ++sample) {
          // Sample context position
          float rand_val = uniform_dist(rng);
          float cumulative = 0.0f;
          int selected_idx = 0;
          
          for (size_t i = 0; i < context_weights.size(); ++i) {
            cumulative += context_weights[i];
            if (rand_val < cumulative) {
              selected_idx = i;
              break;
            }
          }
          
          int context_idx = sentence[context_positions[selected_idx]].vocab_idx;
          float* w_vec = word_emb + context_idx * n_dims;
          std::memset(hidden_errors.data(), 0, n_dims * sizeof(float));
          
          // Negative sampling
          for (int d = 0; d <= n_neg; ++d) {
            int target;
            float label;
            
            if (d == 0) {
              target = word_idx;
              label = 1.0f;
            } else {
              next_random = next_random * 25214903917ULL + 11;
              target = neg_table[(next_random >> 16) % neg_table_size];
              if (target == word_idx) continue;
              label = 0.0f;
            }
            
            float* c_vec = context_emb + target * n_dims;
            
            float dot = 0.0f;
            for (int k = 0; k < n_dims; ++k) {
              dot += w_vec[k] * c_vec[k];
            }
            
            float pred = fast_sigmoid(dot, exp_table);
            float grad = (label - pred) * alpha;
            
            for (int k = 0; k < n_dims; ++k) {
              hidden_errors[k] += grad * c_vec[k];
            }
            
            for (int k = 0; k < n_dims; ++k) {
              c_vec[k] += grad * w_vec[k];
            }
          }
          
          for (int k = 0; k < n_dims; ++k) {
            w_vec[k] += hidden_errors[k];
          }
        }
      }
    }
  }
  
  // Final update of processed count
  processed_words.fetch_add(local_word_count, std::memory_order_relaxed);
}

// [[Rcpp::export]]
List sgns_streaming_cpp(
    const List& tokens_list,           // List of integer vectors from tokens object (one per document)
    const IntegerVector& vocab,         // 1-indexed quanteda type indices to include (filtered in R)
    const NumericVector& type_widths,   // Word widths for distance metric
    const int n_dims,                   // Embedding dimensionality
    const int n_neg,                    // Number of negative samples
    const int window,                   // Context window size
    const double lr,                    // Initial learning rate
    const int epochs,                   // Number of training epochs
    const double context_smoothing,     // Smoothing power for negative sampling
    const double subsample,             // Subsampling threshold
    const std::string weights_type,     // "linear", "harmonic", "exponential", "power", "none"
    const double weights_alpha,         // Alpha parameter for exponential/power
    const NumericVector& weights_vec,   // Custom weight vector
    const int weights_mode,             // 0-4 for weight vector indexing
    const bool include_target,          // Include self-context
    const double forward_weight,        // Forward context weight
    const double backward_weight,       // Backward context weight
    const bool clean_distance,          // Use original positions (clean) vs filtered positions (dirty)
    const std::string init_type,        // "uniform" or "normal"
    const int seed,                     // Random seed
    const bool verbose,                 // Verbose output
    const int threads) {                // Number of threads

  int n_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
  if (n_threads < 1) n_threads = 1;
  
  std::mt19937 rng(seed);
  
  // Build vocabulary
  if (verbose) {
    Rcpp::Rcout << "Building vocabulary...\n";
  }
  
  std::vector<int> type_to_vocab;  // Maps quanteda type index to vocab index
  std::vector<int> vocab_indices_ordered;  // Vocab indices in frequency order (1-indexed)
  std::vector<long long> word_counts;
  long long total_words = 0;
  
  build_vocab(tokens_list, vocab, type_to_vocab, vocab_indices_ordered, word_counts, total_words);
  
  int vocab_len = word_counts.size();
  
  if (verbose) {
    Rcpp::Rcout << "Vocabulary size: " << vocab_len << " words\n";
    Rcpp::Rcout << "Total words in corpus: " << total_words << "\n";
  }
  
  // Initialize embeddings
  int emb_size = vocab_len * n_dims;
  std::vector<float> word_emb(emb_size);
  std::vector<float> context_emb(emb_size);
  
  if (init_type == "uniform") {
    std::uniform_real_distribution<float> init_dist(-0.5f / n_dims, 0.5f / n_dims);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
      context_emb[i] = init_dist(rng);
    }
  } else if (init_type == "normal") {
    std::normal_distribution<float> init_dist(0.0f, 0.01f);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
      context_emb[i] = init_dist(rng);
    }
  } else {
    Rcpp::stop("init_type must be 'uniform' or 'normal'");
  }
  
  // Build negative sampling table using word counts from vocabulary
  double total_weight = 0.0;
  std::vector<double> smoothed_counts(vocab_len);
  for (int i = 0; i < vocab_len; ++i) {
    smoothed_counts[i] = std::pow(static_cast<double>(word_counts[i]), context_smoothing);
    total_weight += smoothed_counts[i];
  }
  
  // Fixed-size table for fast sampling (word2vec uses 100M but let's try smaller)
  const int NEG_TABLE_SIZE = 10000000;  // 10M entries (40MB vs 400MB)
  std::vector<int> neg_table(NEG_TABLE_SIZE);
  int table_idx = 0;
  double cumulative = 0.0;
  for (int i = 0; i < vocab_len && table_idx < NEG_TABLE_SIZE; ++i) {
    cumulative += smoothed_counts[i] / total_weight;
    while (table_idx < NEG_TABLE_SIZE && 
           static_cast<double>(table_idx) / NEG_TABLE_SIZE < cumulative) {
      neg_table[table_idx++] = i;
    }
  }
  while (table_idx < NEG_TABLE_SIZE) {
    neg_table[table_idx++] = vocab_len - 1;
  }
  
  // Build exp table
  std::vector<float> exp_table;
  build_exp_table(exp_table);
  
  // Convert type_widths to C++ vector
  std::vector<float> type_widths_vec(type_widths.size());
  for (int i = 0; i < type_widths.size(); ++i) {
    type_widths_vec[i] = static_cast<float>(type_widths[i]);
  }
  
  // Determine if we're using words metric (all widths = 1.0)
  bool use_words_metric = (type_widths.size() == 0 || 
                          std::all_of(type_widths_vec.begin(), type_widths_vec.end(), 
                                     [](float w) { return w == 1.0f; }));
  
  // For clean distance with character metric, store original tokens per document
  std::vector<IntegerVector> original_docs;
  std::vector<int> doc_start_positions;
  
  if (!use_words_metric && clean_distance) {
    original_docs.reserve(tokens_list.size());
    doc_start_positions.reserve(tokens_list.size());
  }
  
  // For dirty distance with character metric, need vocab_to_type mapping
  std::vector<int> vocab_to_type;
  if (!use_words_metric && !clean_distance) {
    vocab_to_type.resize(vocab_len);
    for (int i = 0; i < vocab_len; ++i) {
      vocab_to_type[i] = vocab_indices_ordered[i] - 1;  // Convert to 0-indexed type
    }
  }
  
  if (verbose) {
    Rcpp::Rcout << "Converting tokens to indices...\n";
  }
  
  // Pre-convert all tokens to Token structs with vocab indices and original positions
  std::vector<std::vector<Token>> sentences;
  sentences.reserve(tokens_list.size());
  
  std::mt19937 subsample_rng(seed + 999);
  std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
  
  long long tokens_before_subsample = 0;
  long long tokens_after_subsample = 0;
  
  for (int doc = 0; doc < tokens_list.size(); ++doc) {
    SEXP doc_sexp = tokens_list[doc];
    if (TYPEOF(doc_sexp) != INTSXP) continue;
    
    IntegerVector doc_tokens = as<IntegerVector>(doc_sexp);
    
    // Store original document for clean distance with character metric
    if (!use_words_metric && clean_distance) {
      original_docs.push_back(doc_tokens);
      doc_start_positions.push_back(sentences.size());
    }
    
    std::vector<Token> sentence;
    sentence.reserve(doc_tokens.size());
    
    for (int i = 0; i < doc_tokens.size(); ++i) {
      int type_idx = doc_tokens[i] - 1;  // Convert from 1-indexed to 0-indexed
      if (type_idx < 0 || type_idx >= static_cast<int>(type_to_vocab.size())) continue;
      
      int vocab_idx = type_to_vocab[type_idx];
      if (vocab_idx < 0) continue;  // Not in our vocabulary
      
      tokens_before_subsample++;
      
      // Subsampling (Mikolov et al. 2013)
      if (subsample > 0) {
        float freq = static_cast<float>(word_counts[vocab_idx]) / total_words;
        float keep_prob = std::sqrt(subsample / freq) + subsample / freq;
        if (uniform_dist(subsample_rng) > keep_prob) continue;
      }
      
      sentence.push_back({vocab_idx, i});  // Store both vocab index and original position
      tokens_after_subsample++;
    }
    
    if (!sentence.empty()) {
      sentences.push_back(std::move(sentence));
    }
  }
  
  long long actual_total_words = tokens_after_subsample * epochs;
  
  if (verbose) {
    Rcpp::Rcout << "Tokens after subsampling: " << tokens_after_subsample 
                << " (" << (100.0 * tokens_after_subsample / tokens_before_subsample) 
                << "%)\n";
  }
  
  // Convert weights_vec to C++ vector
  std::vector<float> weights_vec_cpp(weights_vec.size());
  for (int i = 0; i < weights_vec.size(); ++i) {
    weights_vec_cpp[i] = static_cast<float>(weights_vec[i]);
  }
  
  // Determine if we can use fast path (original word2vec)
  // Note: Fast path doesn't check distance, so clean vs dirty doesn't matter
  bool use_fast_path = (weights_type == "none" && 
                        weights_mode == 0 &&
                        !include_target &&
                        forward_weight == 1.0 &&
                        backward_weight == 1.0 &&
                        use_words_metric);
  
  // Training
  if (verbose) {
    Rcpp::Rcout << "Training SGNS with " << n_threads << " threads\n";
    Rcpp::Rcout << "Total training words: " << actual_total_words << "\n";
    if (use_fast_path) {
      Rcpp::Rcout << "Using fast path (standard word2vec)\n";
    } else {
      Rcpp::Rcout << "Using weighted sampling\n";
      if (clean_distance) {
        Rcpp::Rcout << "Using clean distance (original token positions)\n";
      } else {
        Rcpp::Rcout << "Using dirty distance (filtered token positions)\n";
      }
    }
  }
  
  int n_sentences = sentences.size();
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::atomic<long long> processed_words(epoch * tokens_after_subsample);
    
    if (verbose) {
      Rcpp::Rcout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
    }
    
    // Divide sentences among threads
    std::vector<std::thread> threads_vec;
    int sents_per_thread = (n_sentences + n_threads - 1) / n_threads;
    
    for (int t = 0; t < n_threads; ++t) {
      int start_sent = t * sents_per_thread;
      int end_sent = std::min(start_sent + sents_per_thread, n_sentences);
      if (start_sent >= n_sentences) break;
      
      threads_vec.emplace_back(
        sgns_streaming_worker,
        std::cref(sentences),
        std::cref(word_counts),
        std::cref(neg_table),
        NEG_TABLE_SIZE,
        std::cref(exp_table),
        std::cref(type_widths_vec),
        std::cref(vocab_to_type),
        std::cref(original_docs),
        std::cref(doc_start_positions),
        word_emb.data(),
        context_emb.data(),
        n_dims,
        n_neg,
        window,
        static_cast<float>(lr),
        actual_total_words,
        t,
        n_threads,
        start_sent,
        end_sent,
        std::cref(weights_type),
        static_cast<float>(weights_alpha),
        std::cref(weights_vec_cpp),
        weights_mode,
        include_target,
        static_cast<float>(forward_weight),
        static_cast<float>(backward_weight),
        use_fast_path,
        use_words_metric,
        clean_distance,
        std::ref(processed_words)
      );
    }
    
    // Wait for all threads
    for (auto& t : threads_vec) {
      t.join();
    }
    
    if (verbose) {
      // Calculate final alpha
      float final_alpha = static_cast<float>(lr) * (1.0f - static_cast<float>(processed_words.load()) / actual_total_words);
      if (final_alpha < static_cast<float>(lr) * 0.0001f) final_alpha = static_cast<float>(lr) * 0.0001f;
      Rcpp::Rcout << "  Alpha: " << final_alpha << "\n";
    }
    
    Rcpp::checkUserInterrupt();
  }
  
  // Convert back to R matrices (R will add rownames)
  NumericMatrix word_embeddings(vocab_len, n_dims);
  NumericMatrix context_embeddings(vocab_len, n_dims);
  
  for (int i = 0; i < vocab_len; ++i) {
    for (int d = 0; d < n_dims; ++d) {
      word_embeddings(i, d) = word_emb[i * n_dims + d];
      context_embeddings(i, d) = context_emb[i * n_dims + d];
    }
  }
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings,
    Named("vocab_indices") = vocab_indices_ordered
  );
}

// Thread worker function for FCM-based method - processes a range of training examples
void sgns_thread_worker(
    // Data (shared, read-only)
    const int* word_ids,
    const int* context_ids,
    const float* counts,
    const int* neg_table,
    const int neg_table_size,
    const std::vector<float>& exp_table,
    // Embeddings (shared, read-write - lock-free updates)
    float* word_emb,
    float* context_emb,
    // Parameters
    const int n_dims,
    const int n_neg,
    const float initial_lr,
    const int total_examples,
    const int thread_id,
    const int n_threads,
    // Progress tracking
    std::atomic<long long>& processed_count,
    const long long total_train_words,
    std::atomic<float>& current_alpha
) {
  // Thread-local RNG
  std::mt19937 rng(thread_id * 12345 + 1);
  std::uniform_int_distribution<int> neg_sampler(0, neg_table_size - 1);
  
  // Thread-local gradient accumulator
  std::vector<float> hidden_errors(n_dims);
  
  // Calculate this thread's range
  int examples_per_thread = (total_examples + n_threads - 1) / n_threads;
  int start_idx = thread_id * examples_per_thread;
  int end_idx = std::min(start_idx + examples_per_thread, total_examples);
  
  // For learning rate updates - update every ~10000 words
  const int lr_update_interval = 10000;
  long long local_word_count = 0;
  long long last_word_count = 0;
  
  for (int idx = start_idx; idx < end_idx; ++idx) {
    int word_id = word_ids[idx];
    int context_id = context_ids[idx];
    float count = counts[idx];
    
    // Number of training iterations for this pair
    // Use ceiling of count (standard approach for FCM-based training)
    int n_iters = static_cast<int>(count + 0.5f);
    if (n_iters < 1) n_iters = 1;
    
    for (int iter = 0; iter < n_iters; ++iter) {
      local_word_count++;
      
      // Update learning rate periodically
      if (local_word_count - last_word_count > lr_update_interval) {
        processed_count += (local_word_count - last_word_count);
        last_word_count = local_word_count;
        
        float progress = static_cast<float>(processed_count.load()) / total_train_words;
        float alpha = initial_lr * (1.0f - progress);
        if (alpha < initial_lr * 0.0001f) alpha = initial_lr * 0.0001f;
        current_alpha.store(alpha);
      }
      
      float alpha = current_alpha.load();
      
      // Get pointers to embeddings
      float* w_vec = word_emb + word_id * n_dims;
      
      // Zero out hidden errors
      std::memset(hidden_errors.data(), 0, n_dims * sizeof(float));
      
      // Process positive sample and negative samples together
      // This is the key optimization from word2vec: accumulate gradients
      for (int d = 0; d <= n_neg; ++d) {
        int target;
        float label;
        
        if (d == 0) {
          // Positive sample
          target = context_id;
          label = 1.0f;
        } else {
          // Negative sample
          target = neg_table[neg_sampler(rng)];
          if (target == context_id) continue;  // Skip if same as positive
          label = 0.0f;
        }
        
        float* c_vec = context_emb + target * n_dims;
        
        // Compute dot product
        float dot = 0.0f;
        for (int k = 0; k < n_dims; ++k) {
          dot += w_vec[k] * c_vec[k];
        }
        
        // Compute gradient
        float pred = fast_sigmoid(dot, exp_table);
        float grad = (label - pred) * alpha;
        
        // Accumulate gradient for word vector
        for (int k = 0; k < n_dims; ++k) {
          hidden_errors[k] += grad * c_vec[k];
        }
        
        // Update context vector immediately
        for (int k = 0; k < n_dims; ++k) {
          c_vec[k] += grad * w_vec[k];
        }
      }
      
      // Apply accumulated gradient to word vector
      for (int k = 0; k < n_dims; ++k) {
        w_vec[k] += hidden_errors[k];
      }
    }
  }
  
  // Final update of processed count
  processed_count += (local_word_count - last_word_count);
}

// [[Rcpp::export(rng = false)]]
List sgns_from_fcm_cpp(
    const IntegerVector& i_indices,      // 0-indexed row indices
    const IntegerVector& j_indices,      // 0-indexed column indices
    const NumericVector& x_values,       // co-occurrence counts
    const int n_words,                   // number of words (rows)
    const int n_contexts,                // number of contexts (columns)
    const int n_dims,                    // embedding dimensionality
    const int n_neg,                     // number of negative samples
    const double lr,                     // initial learning rate
    const int epochs,                    // number of training epochs
    const std::string init_type,         // "uniform" or "normal" initialization
    const int seed,                      // random seed
    const bool verbose,                  // verbose output?
    const int threads) {                 // number of threads

  int n_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
  if (n_threads < 1) n_threads = 1;
  
  // Initialize random number generator
  std::mt19937 rng(seed);
  
  // Use float for embeddings (like word2vec)
  int emb_size = n_words * n_dims;
  std::vector<float> word_emb(emb_size);
  std::vector<float> context_emb(emb_size);  // Same size as word_emb (symmetric)
  
  // Initialize embeddings
  if (init_type == "uniform") {
    std::uniform_real_distribution<float> init_dist(-0.5f / n_dims, 0.5f / n_dims);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
      context_emb[i] = init_dist(rng);
    }
  } else if (init_type == "normal") {
    std::normal_distribution<float> init_dist(0.0f, 0.01f);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
      context_emb[i] = init_dist(rng);
    }
  } else {
    Rcpp::stop("init_type must be 'uniform' or 'normal'");
  }
  
  // Build negative sampling table from word frequencies (0.75 power like word2vec)
  std::vector<double> word_counts(n_words, 0.0);
  for (int i = 0; i < x_values.size(); ++i) {
    word_counts[i_indices[i]] += x_values[i];
  }
  
  double total_weight = 0.0;
  for (int i = 0; i < n_words; ++i) {
    word_counts[i] = std::pow(word_counts[i], 0.75);
    total_weight += word_counts[i];
  }
  
  // Fixed-size table for fast sampling (like word2vec)
  const int NEG_TABLE_SIZE = 100000000;  // 100M entries
  std::vector<int> neg_table(NEG_TABLE_SIZE);
  int table_idx = 0;
  double cumulative = 0.0;
  for (int i = 0; i < n_words && table_idx < NEG_TABLE_SIZE; ++i) {
    cumulative += word_counts[i] / total_weight;
    while (table_idx < NEG_TABLE_SIZE && 
           static_cast<double>(table_idx) / NEG_TABLE_SIZE < cumulative) {
      neg_table[table_idx++] = i;
    }
  }
  // Fill remainder with last word
  while (table_idx < NEG_TABLE_SIZE) {
    neg_table[table_idx++] = n_words - 1;
  }
  
  // Build exp table for fast sigmoid
  std::vector<float> exp_table;
  build_exp_table(exp_table);
  
  // Sample training pairs from FCM (stochastic sampling to mimic streaming)
  // Total samples = sum of counts (rounded)
  long long total_train_words = 0;
  for (int i = 0; i < x_values.size(); ++i) {
    total_train_words += static_cast<long long>(x_values[i] + 0.5);
  }
  total_train_words *= epochs;
  
  // Build cumulative distribution for sampling pairs
  int n_pairs = i_indices.size();
  std::vector<double> pair_weights(n_pairs);
  for (int i = 0; i < n_pairs; ++i) {
    pair_weights[i] = x_values[i];
  }
  
  // Create discrete distribution for sampling
  std::discrete_distribution<int> pair_sampler(pair_weights.begin(), pair_weights.end());
  
  // Sample training pairs
  long long n_samples = total_train_words / epochs;
  std::vector<int> sampled_contexts(n_samples);
  std::vector<int> sampled_words(n_samples);
  
  for (long long i = 0; i < n_samples; ++i) {
    int pair_idx = pair_sampler(rng);
    sampled_contexts[i] = j_indices[pair_idx];
    sampled_words[i] = i_indices[pair_idx];
  }
  
  if (verbose) {
    Rcpp::Rcout << "Training SGNS from FCM with " << n_threads << " threads\n";
    Rcpp::Rcout << "Vocabulary size: " << n_words << "\n";
    Rcpp::Rcout << "Total training words: " << total_train_words << "\n";
  }
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Shuffle sampled pairs each epoch
    std::vector<int> shuffle_idx(n_samples);
    std::iota(shuffle_idx.begin(), shuffle_idx.end(), 0);
    std::shuffle(shuffle_idx.begin(), shuffle_idx.end(), rng);
    
    std::vector<int> shuffled_words(n_samples);
    std::vector<int> shuffled_contexts(n_samples);
    for (long long i = 0; i < n_samples; ++i) {
      shuffled_words[i] = sampled_words[shuffle_idx[i]];
      shuffled_contexts[i] = sampled_contexts[shuffle_idx[i]];
    }
    
    // Progress tracking
    std::atomic<long long> processed_count(epoch * n_samples);
    std::atomic<float> current_alpha(static_cast<float>(lr));
    
    if (verbose) {
      Rcpp::Rcout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
    }
    
    // Thread-local RNGs and gradient accumulators
    std::vector<std::thread> threads_vec;
    long long samples_per_thread = (n_samples + n_threads - 1) / n_threads;
    
    for (int t = 0; t < n_threads; ++t) {
      long long start_idx = t * samples_per_thread;
      long long end_idx = std::min(start_idx + samples_per_thread, n_samples);
      
      threads_vec.emplace_back([&, t, start_idx, end_idx]() {
        // Thread-local RNG using LCG like word2vec
        unsigned long long next_random = static_cast<unsigned long long>(seed) * (t + 1);
        
        std::vector<float> hidden_errors(n_dims, 0.0f);
        long long local_word_count = 0;
        float alpha = static_cast<float>(lr);
        
        for (long long idx = start_idx; idx < end_idx; ++idx) {
          int context_word = shuffled_contexts[idx];
          int target_word = shuffled_words[idx];
          
          local_word_count++;
          
          // Update learning rate every 10000 words
          if (local_word_count % 10000 == 0) {
            long long global_count = processed_count.load(std::memory_order_relaxed);
            alpha = static_cast<float>(lr) * (1.0f - static_cast<float>(global_count) / total_train_words);
            if (alpha < static_cast<float>(lr) * 0.0001f) {
              alpha = static_cast<float>(lr) * 0.0001f;
            }
            current_alpha.store(alpha, std::memory_order_relaxed);
          }
          
          // Get context word embedding (this is the input)
          float* w_vec = word_emb.data() + context_word * n_dims;
          
          // Zero hidden errors
          std::memset(hidden_errors.data(), 0, n_dims * sizeof(float));
          
          // Negative sampling
          for (int d = 0; d <= n_neg; ++d) {
            int target;
            float label;
            
            if (d == 0) {
              // Positive sample
              target = target_word;
              label = 1.0f;
            } else {
              // Negative sample
              next_random = next_random * 25214903917ULL + 11;
              target = neg_table[(next_random >> 16) % NEG_TABLE_SIZE];
              if (target == target_word) continue;
              label = 0.0f;
            }
            
            float* c_vec = context_emb.data() + target * n_dims;
            
            // Compute dot product
            float dot = 0.0f;
            for (int k = 0; k < n_dims; ++k) {
              dot += w_vec[k] * c_vec[k];
            }
            
            // Compute gradient
            float pred = fast_sigmoid(dot, exp_table);
            float grad = (label - pred) * alpha;
            
            // Accumulate gradient for input vector
            for (int k = 0; k < n_dims; ++k) {
              hidden_errors[k] += grad * c_vec[k];
            }
            
            // Update output vector
            for (int k = 0; k < n_dims; ++k) {
              c_vec[k] += grad * w_vec[k];
            }
          }
          
          // Apply accumulated gradient to input vector
          for (int k = 0; k < n_dims; ++k) {
            w_vec[k] += hidden_errors[k];
          }
        }
        
        processed_count.fetch_add(local_word_count, std::memory_order_relaxed);
      });
    }
    
    // Wait for all threads
    for (auto& t : threads_vec) {
      t.join();
    }
    
    if (verbose) {
      Rcpp::Rcout << "  Alpha: " << current_alpha.load() << "\n";
    }
    
    // Check for user interrupt
    Rcpp::checkUserInterrupt();
  }
  
  // Convert back to R matrices
  NumericMatrix word_embeddings(n_words, n_dims);
  NumericMatrix context_embeddings(n_words, n_dims);
  
  for (int i = 0; i < n_words; ++i) {
    for (int d = 0; d < n_dims; ++d) {
      word_embeddings(i, d) = word_emb[i * n_dims + d];
      context_embeddings(i, d) = context_emb[i * n_dims + d];
    }
  }
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings
  );
}
