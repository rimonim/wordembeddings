#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <thread>
#include <atomic>
#include <cstring>

using namespace Rcpp;

// Constants for sigmoid lookup table (from word2vec)
constexpr int EXP_TABLE_SIZE = 1000;
constexpr float MAX_EXP = 6.0f;

// Build exp/sigmoid table like original word2vec
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
void build_vocab(
    const List& tokens_list,
    const CharacterVector& token_types,
    int min_count,
    std::vector<std::string>& index_to_word,
    std::vector<int>& type_to_vocab,  // Maps quanteda type index to our vocab index
    std::vector<long long>& word_counts_out,
    long long& total_words
) {
  int n_types = token_types.size();
  
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
  
  // Collect types meeting min_count, with their counts and original indices
  std::vector<std::tuple<std::string, long long, int>> word_vec;
  word_vec.reserve(n_types);
  
  for (int i = 0; i < n_types; ++i) {
    if (type_counts[i] >= min_count) {
      std::string word = as<std::string>(token_types[i]);
      word_vec.push_back({word, type_counts[i], i});
    }
  }
  
  // Sort by frequency (most frequent first)
  std::sort(word_vec.begin(), word_vec.end(),
    [](const auto& a, const auto& b) {
      return std::get<1>(a) > std::get<1>(b);
    });
  
  // Build mapping from quanteda type index to our vocab index
  type_to_vocab.assign(n_types, -1);  // -1 means not in vocab
  word_counts_out.resize(word_vec.size());
  
  for (size_t vocab_idx = 0; vocab_idx < word_vec.size(); ++vocab_idx) {
    const std::string& word = std::get<0>(word_vec[vocab_idx]);
    long long count = std::get<1>(word_vec[vocab_idx]);
    int type_idx = std::get<2>(word_vec[vocab_idx]);
    
    index_to_word.push_back(word);
    word_counts_out[vocab_idx] = count;
    type_to_vocab[type_idx] = vocab_idx;
  }
}

// Thread worker for streaming SGNS (word2vec style)
void sgns_streaming_worker(
    // Data (shared, read-only)
    const std::vector<std::vector<int>>& sentences,  // Pre-converted to indices
    const std::vector<long long>& word_counts,
    const std::vector<int>& neg_table,
    const int neg_table_size,
    const std::vector<float>& exp_table,
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
    // Progress tracking
    std::atomic<long long>& processed_words
) {
  // Fast RNG (Linear Congruential Generator like word2vec)
  unsigned long long next_random = thread_id + 1;
  
  // Thread-local gradient accumulator
  std::vector<float> hidden_errors(n_dims, 0.0f);
  
  long long local_word_count = 0;
  float alpha = initial_lr;
  
  // Process sentences assigned to this thread
  for (int sent_idx = start_sent; sent_idx < end_sent; ++sent_idx) {
    const std::vector<int>& sentence = sentences[sent_idx];
    
    // Skip-gram training on this sentence
    for (size_t pos = 0; pos < sentence.size(); ++pos) {
      int word_idx = sentence[pos];
      
      local_word_count++;
      
      // Update learning rate every 10000 words
      if (local_word_count % 10000 == 0) {
        long long global_word_count = processed_words.load(std::memory_order_relaxed);
        alpha = initial_lr * (1.0f - static_cast<float>(global_word_count) / total_train_words);
        if (alpha < initial_lr * 0.0001f) alpha = initial_lr * 0.0001f;
      }
      
      // Random window size (word2vec does this for regularization)
      next_random = next_random * 25214903917ULL + 11;
      int actual_window = (next_random >> 16) % window;
      
      // Train on contexts in the window
      for (int a = actual_window; a < window * 2 + 1 - actual_window; ++a) {
        if (a == window) continue;  // Skip the word itself
        
        int c_pos = pos - window + a;
        if (c_pos < 0 || c_pos >= static_cast<int>(sentence.size())) continue;
        
        int context_idx = sentence[c_pos];
        
        float* w_vec = word_emb + context_idx * n_dims;  // Note: word2vec uses context word's vector
        
        // Zero out hidden errors
        std::memset(hidden_errors.data(), 0, n_dims * sizeof(float));
        
        // Negative sampling
        for (int d = 0; d <= n_neg; ++d) {
          int target;
          float label;
          
          if (d == 0) {
            // Positive sample
            target = word_idx;
            label = 1.0f;
          } else {
            // Negative sample
            next_random = next_random * 25214903917ULL + 11;
            target = neg_table[(next_random >> 16) % neg_table_size];
            if (target == word_idx) continue;  // Skip if same as positive
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
  }
  
  // Final update of processed count
  processed_words.fetch_add(local_word_count, std::memory_order_relaxed);
}

// [[Rcpp::export]]
List sgns_streaming_cpp(
    const List& tokens_list,         // List of character vectors (one per document)
    const int min_count,              // Minimum word frequency
    const int vocab_size,             // Maximum vocabulary size (0 = unlimited)
    const int n_dims,                 // Embedding dimensionality
    const int n_neg,                  // Number of negative samples
    const int window,                 // Context window size
    const double lr,                  // Initial learning rate
    const int epochs,                 // Number of training epochs
    const double context_smoothing,   // Smoothing power for negative sampling
    const double subsample,           // Subsampling threshold
    const std::string init_type,      // "uniform" or "normal"
    const int seed,                   // Random seed
    const bool verbose,               // Verbose output
    const int threads) {              // Number of threads

  int n_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
  if (n_threads < 1) n_threads = 1;
  
  std::mt19937 rng(seed);
  
  // Extract types (vocabulary) from tokens object
  CharacterVector token_types;
  if (tokens_list.hasAttribute("types")) {
    token_types = tokens_list.attr("types");
  } else {
    Rcpp::stop("tokens object missing 'types' attribute");
  }
  
  // Build vocabulary
  if (verbose) {
    Rcpp::Rcout << "Building vocabulary...\n";
  }
  
  std::vector<std::string> index_to_word;
  std::vector<int> type_to_vocab;  // Maps quanteda type index to vocab index
  std::vector<long long> word_counts;
  long long total_words = 0;
  
  build_vocab(tokens_list, token_types, min_count, index_to_word, type_to_vocab, 
              word_counts, total_words);
  
  // Limit vocabulary size if requested
  if (vocab_size > 0 && static_cast<int>(index_to_word.size()) > vocab_size) {
    index_to_word.resize(vocab_size);
    
    // Update type_to_vocab mapping: words beyond vocab_size become -1
    for (size_t i = 0; i < type_to_vocab.size(); ++i) {
      if (type_to_vocab[i] >= vocab_size) {
        type_to_vocab[i] = -1;
      }
    }
  }
  
  int vocab_len = index_to_word.size();
  
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
  
  if (verbose) {
    Rcpp::Rcout << "Converting tokens to indices...\n";
  }
  
  // Pre-convert all tokens to vocab indices and apply subsampling
  std::vector<std::vector<int>> sentences;
  sentences.reserve(tokens_list.size());
  
  std::mt19937 subsample_rng(seed + 999);
  std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
  
  long long tokens_before_subsample = 0;
  long long tokens_after_subsample = 0;
  
  for (int doc = 0; doc < tokens_list.size(); ++doc) {
    SEXP doc_sexp = tokens_list[doc];
    if (TYPEOF(doc_sexp) != INTSXP) continue;
    
    IntegerVector doc_tokens = as<IntegerVector>(doc_sexp);
    std::vector<int> sentence;
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
      
      sentence.push_back(vocab_idx);
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
  
  // Training
  if (verbose) {
    Rcpp::Rcout << "Training SGNS with " << n_threads << " threads\n";
    Rcpp::Rcout << "Total training words: " << actual_total_words << "\n";
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
  
  // Convert back to R matrices
  NumericMatrix word_embeddings(vocab_len, n_dims);
  NumericMatrix context_embeddings(vocab_len, n_dims);
  CharacterVector vocab_words(vocab_len);
  
  for (int i = 0; i < vocab_len; ++i) {
    vocab_words[i] = index_to_word[i];
    for (int d = 0; d < n_dims; ++d) {
      word_embeddings(i, d) = word_emb[i * n_dims + d];
      context_embeddings(i, d) = context_emb[i * n_dims + d];
    }
  }
  
  rownames(word_embeddings) = vocab_words;
  rownames(context_embeddings) = vocab_words;
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings,
    Named("vocab") = vocab_words
  );
}
