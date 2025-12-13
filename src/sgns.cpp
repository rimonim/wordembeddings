#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cstring>

using namespace Rcpp;

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

// Thread worker function - processes a range of training examples
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
List sgns_train_cpp(
    const IntegerVector& i_indices,      // 0-indexed row indices
    const IntegerVector& j_indices,      // 0-indexed column indices
    const NumericVector& x_values,       // co-occurrence counts
    const int n_words,                   // number of words (rows)
    const int n_contexts,                // number of contexts (columns)
    const int n_dims,                    // embedding dimensionality
    const int n_neg,                     // number of negative samples
    const double lr,                     // initial learning rate
    const int epochs,                    // number of training epochs
    const int grain_size,                // (unused, kept for API compatibility)
    const double smoothing,              // smoothing power for neg sampling
    const bool reject_positives,         // (unused in new impl, rejection is always on)
    const std::string init_type,         // "uniform" or "normal" initialization
    const bool bootstrap_positive,       // (unused, kept for API compatibility)
    const int seed,                      // random seed
    const bool verbose,                  // verbose output?
    const int threads) {                 // number of threads

  int n_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
  if (n_threads < 1) n_threads = 1;
  
  // Initialize random number generator
  std::mt19937 rng(seed);
  
  // Use float for embeddings (like word2vec) - faster and sufficient precision
  int emb_size = n_words * n_dims;
  int ctx_size = n_contexts * n_dims;
  std::vector<float> word_emb(emb_size);
  std::vector<float> context_emb(ctx_size);
  
  // Initialize embeddings
  if (init_type == "uniform") {
    std::uniform_real_distribution<float> init_dist(-0.5f / n_dims, 0.5f / n_dims);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
    }
    for (int i = 0; i < ctx_size; ++i) {
      context_emb[i] = init_dist(rng);
    }
  } else if (init_type == "normal") {
    std::normal_distribution<float> init_dist(0.0f, 0.01f);
    for (int i = 0; i < emb_size; ++i) {
      word_emb[i] = init_dist(rng);
    }
    for (int i = 0; i < ctx_size; ++i) {
      context_emb[i] = init_dist(rng);
    }
  } else {
    Rcpp::stop("init_type must be 'uniform' or 'normal'");
  }
  
  // Build negative sampling table (unigram distribution raised to 0.75 power)
  std::vector<double> context_counts(n_contexts, 0.0);
  for (int i = 0; i < x_values.size(); ++i) {
    context_counts[j_indices[i]] += x_values[i];
  }
  
  double total_weight = 0.0;
  for (int i = 0; i < n_contexts; ++i) {
    context_counts[i] = std::pow(context_counts[i], smoothing);
    total_weight += context_counts[i];
  }
  
  // Fixed-size table for fast sampling (like word2vec)
  const int NEG_TABLE_SIZE = 100000000;  // 100M entries
  std::vector<int> neg_table(NEG_TABLE_SIZE);
  int table_idx = 0;
  double cumulative = 0.0;
  for (int i = 0; i < n_contexts && table_idx < NEG_TABLE_SIZE; ++i) {
    cumulative += context_counts[i] / total_weight;
    while (table_idx < NEG_TABLE_SIZE && 
           static_cast<double>(table_idx) / NEG_TABLE_SIZE < cumulative) {
      neg_table[table_idx++] = i;
    }
  }
  // Fill remainder with last word
  while (table_idx < NEG_TABLE_SIZE) {
    neg_table[table_idx++] = n_contexts - 1;
  }
  
  // Build exp table for fast sigmoid
  std::vector<float> exp_table;
  build_exp_table(exp_table);
  
  // Prepare training data - expand FCM to training examples
  // Shuffle once and reuse (shuffling per epoch is expensive)
  int n_pairs = i_indices.size();
  std::vector<int> word_ids(n_pairs);
  std::vector<int> context_ids(n_pairs);
  std::vector<float> counts(n_pairs);
  
  for (int i = 0; i < n_pairs; ++i) {
    word_ids[i] = i_indices[i];
    context_ids[i] = j_indices[i];
    counts[i] = static_cast<float>(x_values[i]);
  }
  
  // Calculate total training words (for learning rate schedule)
  long long total_train_words = 0;
  for (int i = 0; i < n_pairs; ++i) {
    total_train_words += static_cast<long long>(counts[i] + 0.5f);
  }
  total_train_words *= epochs;
  
  if (verbose) {
    Rcpp::Rcout << "Training SGNS with " << n_threads << " threads\n";
    Rcpp::Rcout << "Vocabulary: " << n_words << " words, " << n_contexts << " contexts\n";
    Rcpp::Rcout << "Total training examples: " << total_train_words << "\n";
  }
  
  std::vector<double> loss_history;
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Shuffle training data
    std::vector<int> shuffle_idx(n_pairs);
    std::iota(shuffle_idx.begin(), shuffle_idx.end(), 0);
    std::shuffle(shuffle_idx.begin(), shuffle_idx.end(), rng);
    
    std::vector<int> shuffled_words(n_pairs);
    std::vector<int> shuffled_contexts(n_pairs);
    std::vector<float> shuffled_counts(n_pairs);
    for (int i = 0; i < n_pairs; ++i) {
      shuffled_words[i] = word_ids[shuffle_idx[i]];
      shuffled_contexts[i] = context_ids[shuffle_idx[i]];
      shuffled_counts[i] = counts[shuffle_idx[i]];
    }
    
    // Progress tracking
    std::atomic<long long> processed_count(epoch * (total_train_words / epochs));
    std::atomic<float> current_alpha(static_cast<float>(lr));
    
    if (verbose) {
      Rcpp::Rcout << "Epoch " << (epoch + 1) << "/" << epochs << "\n";
    }
    
    // Launch worker threads
    std::vector<std::thread> threads_vec;
    for (int t = 0; t < n_threads; ++t) {
      threads_vec.emplace_back(
        sgns_thread_worker,
        shuffled_words.data(),
        shuffled_contexts.data(),
        shuffled_counts.data(),
        neg_table.data(),
        NEG_TABLE_SIZE,
        std::cref(exp_table),
        word_emb.data(),
        context_emb.data(),
        n_dims,
        n_neg,
        static_cast<float>(lr),
        n_pairs,
        t,
        n_threads,
        std::ref(processed_count),
        total_train_words,
        std::ref(current_alpha)
      );
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
    
    loss_history.push_back(0.0);  // Loss tracking removed for performance
  }
  
  // Convert back to R matrices
  NumericMatrix word_embeddings(n_words, n_dims);
  NumericMatrix context_embeddings(n_contexts, n_dims);
  
  for (int i = 0; i < n_words; ++i) {
    for (int d = 0; d < n_dims; ++d) {
      word_embeddings(i, d) = word_emb[i * n_dims + d];
    }
  }
  for (int i = 0; i < n_contexts; ++i) {
    for (int d = 0; d < n_dims; ++d) {
      context_embeddings(i, d) = context_emb[i * n_dims + d];
    }
  }
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings,
    Named("loss_history") = loss_history
  );
}
