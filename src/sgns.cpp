#include <Rcpp.h>
#include <RcppParallel.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>

using namespace Rcpp;
using namespace RcppParallel;

// [[Rcpp::depends(RcppParallel)]]

struct SGNSWorker : public Worker {
  // Inputs
  const RVector<int> i_indices;
  const RVector<int> j_indices;
  const RVector<double> x_values;
  const std::vector<int>& pair_indices;
  const std::vector<int>& neg_table;
  const std::vector<double>& sigmoid_table;
  
  // Outputs
  RMatrix<double> word_embeddings;
  RMatrix<double> context_embeddings;
  
  // Parameters
  const int n_dims;
  const int n_neg;
  const double lr;
  const int n_iterations;
  const int n_pairs;
  const int iter;
  const double smoothing;
  const bool reject_positives;
  const bool bootstrap_positive;
  const int seed;
  
  // Constants
  const double MAX_SIGMOID = 8.0;
  const int SIGMOID_TABLE_SIZE = 512;

  SGNSWorker(
    const IntegerVector& i_indices,
    const IntegerVector& j_indices,
    const NumericVector& x_values,
    const std::vector<int>& pair_indices,
    const std::vector<int>& neg_table,
    const std::vector<double>& sigmoid_table,
    NumericMatrix& word_embeddings,
    NumericMatrix& context_embeddings,
    int n_dims, int n_neg, double lr, int n_iterations, int n_pairs, int iter,
    double smoothing, bool reject_positives, bool bootstrap_positive, int seed
  ) : i_indices(i_indices), j_indices(j_indices), x_values(x_values),
      pair_indices(pair_indices), neg_table(neg_table), sigmoid_table(sigmoid_table),
      word_embeddings(word_embeddings), context_embeddings(context_embeddings),
      n_dims(n_dims), n_neg(n_neg), lr(lr), n_iterations(n_iterations),
      n_pairs(n_pairs), iter(iter), smoothing(smoothing),
      reject_positives(reject_positives), bootstrap_positive(bootstrap_positive),
      seed(seed) {}
      
  double get_sigmoid(double x) {
    if (x < -MAX_SIGMOID) return 0.0;
    if (x > MAX_SIGMOID) return 1.0;
    
    int idx = static_cast<int>(((x + MAX_SIGMOID) / (2.0 * MAX_SIGMOID)) * 512.0);
    idx = std::max(0, std::min(512, idx));
    return sigmoid_table[idx];
  }

  void operator()(std::size_t begin, std::size_t end) {
    // Initialize thread-local RNG
    // Use seed + iter + begin to ensure different seeds across threads and iterations
    std::mt19937 rng(seed + iter * 1000 + begin);
    std::uniform_int_distribution<int> neg_sampler(0, neg_table.size() - 1);
    
    // Temporary vectors to avoid allocation in loop
    std::vector<double> w(n_dims), c(n_dims), n_vec(n_dims);
    
    for (std::size_t batch_idx = begin; batch_idx < end; ++batch_idx) {
      // Calculate learning rate
      double progress = static_cast<double>(iter * n_pairs + batch_idx) / 
                        (static_cast<double>(n_iterations) * n_pairs);
      double current_lr = lr * (1.0 - progress);
      if (current_lr < 0.0001) current_lr = 0.0001; // Minimum LR
      
      int pair_idx = pair_indices[batch_idx];
      int word_id = i_indices[pair_idx];
      int context_id = j_indices[pair_idx];
      double count = x_values[pair_idx];
      
      // Determine number of training events
      int n_events;
      if (bootstrap_positive) {
        std::poisson_distribution<int> poisson_dist(count);
        n_events = poisson_dist(rng);
      } else {
        n_events = static_cast<int>(std::ceil(count));
      }
      
      for (int event = 0; event < n_events; ++event) {
        // Read vectors
        for (int d = 0; d < n_dims; ++d) {
          w[d] = word_embeddings(word_id, d);
          c[d] = context_embeddings(context_id, d);
        }
        
        // Positive sample
        double dot_pos = 0.0;
        for (int d = 0; d < n_dims; ++d) {
          dot_pos += w[d] * c[d];
        }
        
        double pred_pos = get_sigmoid(dot_pos);
        double grad_pos = (1.0 - pred_pos) * current_lr;
        
        // Update for positive
        for (int d = 0; d < n_dims; ++d) {
          word_embeddings(word_id, d) += grad_pos * c[d];
          context_embeddings(context_id, d) += grad_pos * w[d];
        }
        
        // Negative samples
        for (int n = 0; n < n_neg; ++n) {
          int neg_context_id;
          if (reject_positives) {
            do {
              neg_context_id = neg_table[neg_sampler(rng)];
            } while (neg_context_id == context_id);
          } else {
            neg_context_id = neg_table[neg_sampler(rng)];
          }
          
          for (int d = 0; d < n_dims; ++d) {
            n_vec[d] = context_embeddings(neg_context_id, d);
          }
          
          double dot_neg = 0.0;
          for (int d = 0; d < n_dims; ++d) {
            dot_neg += w[d] * n_vec[d]; // Note: using original w here is standard approximation
          }
          
          double pred_neg = get_sigmoid(dot_neg);
          double grad_neg = -pred_neg * current_lr;
          
          for (int d = 0; d < n_dims; ++d) {
            word_embeddings(word_id, d) += grad_neg * n_vec[d];
            context_embeddings(neg_context_id, d) += grad_neg * w[d];
          }
        }
      }
    }
  }
};

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
    const int batch_size,                // batch size (used for grain size in parallel)
    const double smoothing,              // smoothing power for neg sampling
    const bool reject_positives,         // reject positive context in neg sampling?
    const std::string init_type,         // "uniform" or "normal" initialization
    const bool bootstrap_positive,       // bootstrap positive samples?
    const int seed,                      // random seed
    const bool verbose) {                // verbose output?

  // Initialize random number generator
  std::mt19937 rng(seed);
  
  // Initialize embeddings based on init_type
  NumericMatrix word_embeddings(n_words, n_dims);
  NumericMatrix context_embeddings(n_contexts, n_dims);
  
  if (init_type == "uniform") {
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
    std::vector<double> context_counts(n_contexts, 0.0);
    for (int i = 0; i < x_values.size(); ++i) {
      context_counts[j_indices[i]] += x_values[i];
    }
    
    double total_weight = 0.0;
    for (int i = 0; i < n_contexts; ++i) {
      weights[i] = std::pow(context_counts[i], smoothing);
      total_weight += weights[i];
    }
    
    for (int i = 0; i < n_contexts; ++i) {
      weights[i] /= total_weight;
    }
    
    int table_size = static_cast<int>(std::min(100000000.0, 
                      std::max(1000.0, 100.0 * n_contexts)));
    
    for (int i = 0; i < n_contexts; ++i) {
      int count = std::max(1, static_cast<int>(std::round(weights[i] * table_size)));
      for (int j = 0; j < count; ++j) {
        neg_table.push_back(i);
      }
    }
    
    if (neg_table.size() > table_size) {
      neg_table.resize(table_size);
    }
  }
  
  // Precompute sigmoid lookup table
  const int SIGMOID_TABLE_SIZE = 512;
  const double MAX_SIGMOID = 8.0;
  std::vector<double> sigmoid_table(SIGMOID_TABLE_SIZE + 1);
  for (int i = 0; i <= SIGMOID_TABLE_SIZE; ++i) {
    double x = (2.0 * i / SIGMOID_TABLE_SIZE - 1.0) * MAX_SIGMOID;
    sigmoid_table[i] = 1.0 / (1.0 + std::exp(-x));
  }
  
  // Create shuffled indices for co-occurrence data
  std::vector<int> pair_indices(i_indices.size());
  std::iota(pair_indices.begin(), pair_indices.end(), 0);
  
  int n_pairs = i_indices.size();
  
  for (int iter = 0; iter < n_iterations; ++iter) {
    if (verbose) {
      Rcpp::Rcout << "Iteration " << (iter + 1) << "/" << n_iterations << "\n";
    }
    
    // Shuffle pairs
    std::shuffle(pair_indices.begin(), pair_indices.end(), rng);
    
    // Run parallel training
    SGNSWorker worker(
      i_indices, j_indices, x_values, pair_indices, neg_table, sigmoid_table,
      word_embeddings, context_embeddings,
      n_dims, n_neg, lr, n_iterations, n_pairs, iter,
      smoothing, reject_positives, bootstrap_positive, seed
    );
    
    // Use batch_size as grain size if it's significantly larger than 1
    // Otherwise let RcppParallel decide the grain size
    if (batch_size > 100) {
      parallelFor(0, n_pairs, worker, batch_size);
    } else {
      parallelFor(0, n_pairs, worker);
    }
    
    // Check for user interrupt
    Rcpp::checkUserInterrupt();
  }
  
  return List::create(
    Named("word_embeddings") = word_embeddings,
    Named("context_embeddings") = context_embeddings
  );
}
