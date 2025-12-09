#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>

using namespace Rcpp;

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
    const int n_iterations,              // number of training iterations
    const int batch_size,                // batch size for updates
    const double smoothing,              // smoothing power for neg sampling
    const bool reject_positives,         // reject positive context in neg sampling?
    const std::string init_type,         // "uniform" or "normal" initialization
    const bool bootstrap_positive,       // bootstrap positive samples?
    const int seed,                      // random seed
    const bool verbose) {                // verbose output?

  // Initialize random number generator
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  
  // Initialize embeddings based on init_type
  NumericMatrix word_embeddings(n_words, n_dims);
  NumericMatrix context_embeddings(n_contexts, n_dims);
  
  if (init_type == "uniform") {
    // Uniform distribution U(-0.5/n_dims, 0.5/n_dims)
    std::uniform_real_distribution<double> init_dist(-0.5 / n_dims, 0.5 / n_dims);
    for (int i = 0; i < n_words; ++i) {
      for (int d = 0; d < n_dims; ++d) {
        word_embeddings(i, d) = init_dist(rng);
      }
    }
    for (int i = 0; i < n_contexts; ++i) {
      for (int d = 0; d < n_dims; ++d) {
        context_embeddings(i, d) = init_dist(rng);
      }
    }
  } else if (init_type == "normal") {
    // Standard normal distribution N(0, 1)
    std::normal_distribution<double> init_dist(0.0, 1.0);
    for (int i = 0; i < n_words; ++i) {
      for (int d = 0; d < n_dims; ++d) {
        word_embeddings(i, d) = init_dist(rng);
      }
    }
    for (int i = 0; i < n_contexts; ++i) {
      for (int d = 0; d < n_dims; ++d) {
        context_embeddings(i, d) = init_dist(rng);
      }
    }
  } else {
    Rcpp::stop("init_type must be 'uniform' or 'normal'");
  }
  
  // Build negative sampling lookup table
  std::vector<int> neg_table;
  {
    std::vector<double> weights(n_contexts);
    
    // Frequency-weighted: P(w) ∝ count(w)^smoothing
    // First sum counts for each context
    std::vector<double> context_counts(n_contexts, 0.0);
    for (int i = 0; i < x_values.size(); ++i) {
      context_counts[j_indices[i]] += x_values[i];
    }
    
    // Compute weights
    double total_weight = 0.0;
    for (int i = 0; i < n_contexts; ++i) {
      weights[i] = std::pow(context_counts[i], smoothing);
      total_weight += weights[i];
    }
    
    // Normalize
    for (int i = 0; i < n_contexts; ++i) {
      weights[i] /= total_weight;
    }
    
    // Build lookup table
    // Target size: min(1e8, max(1000, 100*n_contexts))
    int table_size = static_cast<int>(std::min(100000000.0, 
                      std::max(1000.0, 100.0 * n_contexts)));
    
    for (int i = 0; i < n_contexts; ++i) {
      int count = std::max(1, static_cast<int>(std::round(weights[i] * table_size)));
      for (int j = 0; j < count; ++j) {
        neg_table.push_back(i);
      }
    }
    
    // Trim to size if needed
    if (neg_table.size() > table_size) {
      neg_table.resize(table_size);
    }
  }
  
  std::uniform_int_distribution<int> neg_sampler(0, neg_table.size() - 1);
  
  // Create shuffled indices for co-occurrence data
  std::vector<int> pair_indices(i_indices.size());
  std::iota(pair_indices.begin(), pair_indices.end(), 0);
  
  // Precompute sigmoid lookup table for speed
  const int SIGMOID_TABLE_SIZE = 512;
  const double MAX_SIGMOID = 8.0;
  std::vector<double> sigmoid_table(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
    double x = (2.0 * i / SIGMOID_TABLE_SIZE - 1.0) * MAX_SIGMOID;
    sigmoid_table[i] = 1.0 / (1.0 + std::exp(-x));
  }
  
  auto sigmoid = [&sigmoid_table, MAX_SIGMOID](double x) -> double {
    if (x < -MAX_SIGMOID) return 0.0;
    if (x > MAX_SIGMOID) return 1.0;
    
    int idx = static_cast<int>(((x + MAX_SIGMOID) / (2.0 * MAX_SIGMOID)) * 512.0);
    idx = std::max(0, std::min(512, idx));
    return sigmoid_table[idx];
  };
  
  // Main training loop
  double total_loss = 0.0;
  int n_pairs = i_indices.size();
  
  for (int iter = 0; iter < n_iterations; ++iter) {
    if (verbose) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iterations << ": ";
    }
    
    // Shuffle pairs
    std::shuffle(pair_indices.begin(), pair_indices.end(), rng);
    
    // Process batches
    double iter_loss = 0.0;
    
    for (int batch_start = 0; batch_start < n_pairs; batch_start += batch_size) {
      int batch_end = std::min(batch_start + batch_size, n_pairs);
      
      // Current learning rate with linear decay
      double progress = static_cast<double>(iter * n_pairs + batch_start) / 
                        (n_iterations * n_pairs);
      double current_lr = lr * (1.0 - progress);
      
      // Process each co-occurrence in batch
      for (int batch_idx = batch_start; batch_idx < batch_end; ++batch_idx) {
        int pair_idx = pair_indices[batch_idx];
        
        int word_id = i_indices[pair_idx];
        int context_id = j_indices[pair_idx];
        double count = x_values[pair_idx];
        
        // Determine number of training events
        int n_events;
        if (bootstrap_positive) {
          // Poisson sample
          std::poisson_distribution<int> poisson_dist(count);
          n_events = poisson_dist(rng);
        } else {
          n_events = static_cast<int>(std::ceil(count));
        }
        
        // Update for each event
        for (int event = 0; event < n_events; ++event) {
          // Extract vectors
          std::vector<double> w(n_dims), c(n_dims);
          for (int d = 0; d < n_dims; ++d) {
            w[d] = word_embeddings(word_id, d);
            c[d] = context_embeddings(context_id, d);
          }
          
          // Positive sample: maximize log σ(w·c)
          double dot_pos = 0.0;
          for (int d = 0; d < n_dims; ++d) {
            dot_pos += w[d] * c[d];
          }
          
          double pred_pos = sigmoid(dot_pos);
          double loss_pos = -std::log(std::max(pred_pos, 1e-8));
          iter_loss += loss_pos;
          
          // Gradient for positive: (1 - σ(w·c))
          double grad_pos = (1.0 - pred_pos) * current_lr;
          for (int d = 0; d < n_dims; ++d) {
            word_embeddings(word_id, d) += grad_pos * c[d];
            context_embeddings(context_id, d) += grad_pos * w[d];
          }
          
          // Negative samples
          for (int n = 0; n < n_neg; ++n) {
            // Sample negative context
            int neg_context_id;
            if (reject_positives) {
              // Rejection sampling: avoid positive context
              do {
                neg_context_id = neg_table[neg_sampler(rng)];
              } while (neg_context_id == context_id);
            } else {
              // Independent sampling: no rejection
              neg_context_id = neg_table[neg_sampler(rng)];
            }
            
            // Extract negative context vector
            std::vector<double> n_vec(n_dims);
            for (int d = 0; d < n_dims; ++d) {
              n_vec[d] = context_embeddings(neg_context_id, d);
            }
            
            // Negative: minimize log σ(w·n)
            double dot_neg = 0.0;
            for (int d = 0; d < n_dims; ++d) {
              dot_neg += w[d] * n_vec[d];
            }
            
            double pred_neg = sigmoid(dot_neg);
            double loss_neg = -std::log(std::max(1.0 - pred_neg, 1e-8));
            iter_loss += loss_neg;
            
            // Gradient for negative: -σ(w·n)
            double grad_neg = -pred_neg * current_lr;
            for (int d = 0; d < n_dims; ++d) {
              word_embeddings(word_id, d) += grad_neg * n_vec[d];
              context_embeddings(neg_context_id, d) += grad_neg * w[d];
            }
          }
        }
      }
    }
    
    total_loss += iter_loss;
    
    if (verbose) {
      double avg_loss = iter_loss / std::accumulate(x_values.begin(), x_values.end(), 0.0);
      Rcpp::Rcout << "loss = " << std::fixed << std::setprecision(6) << avg_loss << "\n";
    }
    
    // Check for user interrupt
    if (iter % 10 == 0) {
      Rcpp::checkUserInterrupt();
    }
  }
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings,
    Named("total_loss") = total_loss
  );
}
