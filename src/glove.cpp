#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Weighting function types
constexpr int WEIGHT_GLOVE = 0;
constexpr int WEIGHT_POWER = 1;
constexpr int WEIGHT_LOG = 2;
constexpr int WEIGHT_UNIFORM = 3;

// Clipping value for numerical stability
constexpr double CLIP_VALUE = 10.0;

// GloVe fitting class
template <typename T>
class GloveFit {
public:
  // Dimensions
  size_t vocab_size;
  size_t word_vec_size;
  
  // Parameters
  T x_max;
  T learning_rate;
  T alpha;
  int weight_type;
  bool fix_bias;
  
  // Word embeddings (word_vec_size x vocab_size) - stored column-major for efficiency
  std::vector<T> w_i;
  std::vector<T> w_j;
  
  // Biases
  std::vector<T> b_i;
  std::vector<T> b_j;
  
  // AdaGrad squared gradient accumulators
  std::vector<T> grad_sq_w_i;
  std::vector<T> grad_sq_w_j;
  std::vector<T> grad_sq_b_i;
  std::vector<T> grad_sq_b_j;
  
  GloveFit(size_t vocab_size_, size_t word_vec_size_, T x_max_, T learning_rate_,
           T alpha_, int weight_type_, bool fix_bias_,
           const std::vector<T>& w_i_init, const std::vector<T>& w_j_init,
           const std::vector<T>& b_i_init, const std::vector<T>& b_j_init)
    : vocab_size(vocab_size_), word_vec_size(word_vec_size_),
      x_max(x_max_), learning_rate(learning_rate_), alpha(alpha_),
      weight_type(weight_type_), fix_bias(fix_bias_),
      w_i(w_i_init), w_j(w_j_init), b_i(b_i_init), b_j(b_j_init) {
    
    // Initialize AdaGrad accumulators to 1.0
    size_t emb_size = vocab_size * word_vec_size;
    grad_sq_w_i.resize(emb_size, 1.0);
    grad_sq_w_j.resize(emb_size, 1.0);
    grad_sq_b_i.resize(vocab_size, 1.0);
    grad_sq_b_j.resize(vocab_size, 1.0);
  }
  
  // Weighting function
  inline T weight_fn(T x) const {
    switch (weight_type) {
      case WEIGHT_GLOVE:
        return (x < x_max) ? std::pow(x / x_max, alpha) : static_cast<T>(1.0);
      case WEIGHT_POWER:
        return std::pow(x, alpha);
      case WEIGHT_LOG:
        return std::min(static_cast<T>(1.0), std::log(x) / std::log(x_max));
      case WEIGHT_UNIFORM:
      default:
        return static_cast<T>(1.0);
    }
  }
  
  // Train one epoch on the FCM
  // Returns the total cost for this epoch
  T train_epoch(const int* row_indices, const int* col_indices, 
                const double* values, size_t nnz,
                const int* shuffle_order, int n_threads) {
    
    T global_cost = 0.0;
    bool do_shuffle = (shuffle_order != nullptr);
    
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(+:global_cost) schedule(static)
#endif
    for (size_t idx = 0; idx < nnz; ++idx) {
      size_t i_order = do_shuffle ? static_cast<size_t>(shuffle_order[idx]) : idx;
      
      // Get indices (0-based)
      size_t i = static_cast<size_t>(row_indices[i_order]);
      size_t j = static_cast<size_t>(col_indices[i_order]);
      T x_val = static_cast<T>(values[i_order]);
      
      // Skip zero or negative values
      if (x_val <= 0) continue;
      
      // Calculate weight
      T weight = weight_fn(x_val);
      
      // Get pointers to word vectors
      T* w_i_ptr = w_i.data() + i * word_vec_size;
      T* w_j_ptr = w_j.data() + j * word_vec_size;
      T* grad_sq_w_i_ptr = grad_sq_w_i.data() + i * word_vec_size;
      T* grad_sq_w_j_ptr = grad_sq_w_j.data() + j * word_vec_size;
      
      // Compute dot product w_i · w_j
      T dot = 0.0;
      for (size_t k = 0; k < word_vec_size; ++k) {
        dot += w_i_ptr[k] * w_j_ptr[k];
      }
      
      // Compute cost_inner = w_i · w_j + b_i + b_j - log(x)
      T cost_inner = dot + b_i[i] + b_j[j] - std::log(x_val);
      
      // Clip for numerical stability
      if (cost_inner > CLIP_VALUE) cost_inner = CLIP_VALUE;
      else if (cost_inner < -CLIP_VALUE) cost_inner = -CLIP_VALUE;
      
      // Weighted cost
      T cost = weight * cost_inner;
      
      // Accumulate squared cost for loss
      global_cost += cost * cost_inner;
      
      // Compute gradients and update with AdaGrad
      // grad = cost * other_vector (for embeddings)
      // grad = cost (for biases)
      
      // Update word vectors
      for (size_t k = 0; k < word_vec_size; ++k) {
        T grad_w_i = cost * w_j_ptr[k];
        T grad_w_j = cost * w_i_ptr[k];
        
        // AdaGrad update
        w_i_ptr[k] -= learning_rate * grad_w_i / std::sqrt(grad_sq_w_i_ptr[k]);
        w_j_ptr[k] -= learning_rate * grad_w_j / std::sqrt(grad_sq_w_j_ptr[k]);
        
        // Update squared gradients
        grad_sq_w_i_ptr[k] += grad_w_i * grad_w_i;
        grad_sq_w_j_ptr[k] += grad_w_j * grad_w_j;
      }
      
      // Update biases (unless fixed)
      if (!fix_bias) {
        T grad_b_i = cost;
        T grad_b_j = cost;
        
        b_i[i] -= learning_rate * grad_b_i / std::sqrt(grad_sq_b_i[i]);
        b_j[j] -= learning_rate * grad_b_j / std::sqrt(grad_sq_b_j[j]);
        
        grad_sq_b_i[i] += grad_b_i * grad_b_i;
        grad_sq_b_j[j] += grad_b_j * grad_b_j;
      }
    }
    
    return 0.5 * global_cost;
  }
};

// [[Rcpp::export]]
List glove_fit_cpp(
    const IntegerVector& i_indices,      // 0-indexed row indices
    const IntegerVector& j_indices,      // 0-indexed column indices
    const NumericVector& x_values,       // co-occurrence counts
    const int n_rows,                    // number of rows (word vocab)
    const int n_cols,                    // number of cols (context vocab)
    const int n_dims,                    // embedding dimensionality
    const double x_max,                  // max co-occurrence for weighting
    const double alpha,                  // weighting exponent
    const double lr,                     // learning rate
    const int epochs,                    // number of training epochs
    const std::string& weight_type_str,  // "glove", "power", "log", "uniform"
    const bool fix_bias,                 // fix biases at log(marginal counts)?
    const NumericVector& row_sums,       // row marginal sums (for fix_bias)
    const NumericVector& col_sums,       // col marginal sums (for fix_bias)
    const std::string& init_type,        // "uniform" or "normal"
    const int seed,
    const bool verbose,
    const bool shuffle,
    const int threads,
    const bool include_word_embeddings,
    const bool include_context_embeddings
) {
  
  int n_threads = threads;
#ifdef _OPENMP
  if (n_threads <= 0) n_threads = omp_get_max_threads();
#else
  n_threads = 1;
#endif
  
  // Parse weight type
  int weight_type = WEIGHT_GLOVE;
  if (weight_type_str == "power") weight_type = WEIGHT_POWER;
  else if (weight_type_str == "log") weight_type = WEIGHT_LOG;
  else if (weight_type_str == "uniform") weight_type = WEIGHT_UNIFORM;
  
  // Initialize random number generator
  std::mt19937 rng(seed);
  
  // Initialize embeddings
  size_t emb_size_rows = static_cast<size_t>(n_rows) * n_dims;
  size_t emb_size_cols = static_cast<size_t>(n_cols) * n_dims;
  
  std::vector<double> w_i(emb_size_rows);
  std::vector<double> w_j(emb_size_cols);
  std::vector<double> b_i(n_rows);
  std::vector<double> b_j(n_cols);
  
  if (init_type == "uniform") {
    std::uniform_real_distribution<double> init_dist(-0.5 / n_dims, 0.5 / n_dims);
    for (size_t k = 0; k < emb_size_rows; ++k) w_i[k] = init_dist(rng);
    for (size_t k = 0; k < emb_size_cols; ++k) w_j[k] = init_dist(rng);
    
    std::uniform_real_distribution<double> bias_dist(-0.5, 0.5);
    if (!fix_bias) {
      for (int k = 0; k < n_rows; ++k) b_i[k] = bias_dist(rng);
      for (int k = 0; k < n_cols; ++k) b_j[k] = bias_dist(rng);
    }
  } else {  // normal
    std::normal_distribution<double> init_dist(0.0, 0.01);
    for (size_t k = 0; k < emb_size_rows; ++k) w_i[k] = init_dist(rng);
    for (size_t k = 0; k < emb_size_cols; ++k) w_j[k] = init_dist(rng);
    
    if (!fix_bias) {
      for (int k = 0; k < n_rows; ++k) b_i[k] = init_dist(rng);
      for (int k = 0; k < n_cols; ++k) b_j[k] = init_dist(rng);
    }
  }
  
  // If fix_bias, set biases to log of marginal sums
  if (fix_bias) {
    for (int k = 0; k < n_rows; ++k) {
      b_i[k] = (row_sums[k] > 0) ? std::log(row_sums[k]) : 0.0;
    }
    for (int k = 0; k < n_cols; ++k) {
      b_j[k] = (col_sums[k] > 0) ? std::log(col_sums[k]) : 0.0;
    }
  }
  
  // Create GloVe fitter
  GloveFit<double> glove(
    n_rows, n_dims, x_max, lr, alpha, weight_type, fix_bias,
    w_i, w_j, b_i, b_j
  );
  
  // Get raw pointers to data
  const int* row_ptr = i_indices.begin();
  const int* col_ptr = j_indices.begin();
  const double* val_ptr = x_values.begin();
  size_t nnz = static_cast<size_t>(x_values.size());
  
  // Shuffle indices if requested
  std::vector<int> shuffle_indices(nnz);
  std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
  
  if (verbose) {
    Rcpp::Rcout << "Training GloVe with " << n_threads << " threads\n";
    Rcpp::Rcout << "Vocabulary size: " << n_rows << " words, " << n_cols << " contexts\n";
    Rcpp::Rcout << "Non-zero entries: " << nnz << "\n";
    Rcpp::Rcout << "Weighting: " << weight_type_str << " (x_max=" << x_max << ", alpha=" << alpha << ")\n";
    if (fix_bias) Rcpp::Rcout << "Biases fixed at log(marginal counts)\n";
  }
  
  // Training loop
  std::vector<double> cost_history(epochs);
  
  for (int epoch = 0; epoch < epochs; ++epoch) {
    // Shuffle if requested
    if (shuffle) {
      std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);
    }
    
    // Train one epoch
    double cost = glove.train_epoch(
      row_ptr, col_ptr, val_ptr, nnz,
      shuffle ? shuffle_indices.data() : nullptr,
      n_threads
    );
    
    cost_history[epoch] = cost / nnz;
    
    if (verbose) {
      Rcpp::Rcout << "Epoch " << (epoch + 1) << "/" << epochs 
                  << ", loss: " << cost_history[epoch] << "\n";
    }
    
    // Check for user interrupt
    Rcpp::checkUserInterrupt();
  }
  
  // Convert embeddings to R matrices
  // GloVe stores as (n_dims x vocab) but we want (vocab x n_dims)
  NumericMatrix word_embeddings;
  NumericMatrix context_embeddings;
  
  if (include_word_embeddings) {
    word_embeddings = NumericMatrix(n_rows, n_dims);
    for (int i = 0; i < n_rows; ++i) {
      for (int d = 0; d < n_dims; ++d) {
        word_embeddings(i, d) = glove.w_i[i * n_dims + d];
      }
    }
  }
  
  if (include_context_embeddings) {
    context_embeddings = NumericMatrix(n_cols, n_dims);
    for (int j = 0; j < n_cols; ++j) {
      for (int d = 0; d < n_dims; ++d) {
        context_embeddings(j, d) = glove.w_j[j * n_dims + d];
      }
    }
  }
  
  // Return results
  List result;
  if (include_word_embeddings) {
    result["word_embeddings"] = word_embeddings;
  }
  if (include_context_embeddings) {
    result["context_embeddings"] = context_embeddings;
  }
  
  // Also return biases
  NumericVector bias_i(n_rows);
  NumericVector bias_j(n_cols);
  for (int k = 0; k < n_rows; ++k) bias_i[k] = glove.b_i[k];
  for (int k = 0; k < n_cols; ++k) bias_j[k] = glove.b_j[k];
  
  result["bias_i"] = bias_i;
  result["bias_j"] = bias_j;
  result["cost_history"] = cost_history;
  
  return result;
}
