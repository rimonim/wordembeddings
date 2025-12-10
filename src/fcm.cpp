#include <Rcpp.h>
#include <RcppParallel.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_arena.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>

using namespace Rcpp;

// Helper to combine two 32-bit ints into one 64-bit key
inline unsigned long long make_key(unsigned int row, unsigned int col) {
    return ((unsigned long long)row << 32) | col;
}

inline std::pair<unsigned int, unsigned int> split_key(unsigned long long key) {
    return std::make_pair((unsigned int)(key >> 32), (unsigned int)(key & 0xFFFFFFFF));
}

typedef std::unordered_map<unsigned long long, double> MapType;

struct FCMBody {
    // Inputs
    const std::vector<const int*>& doc_ptrs;
    const std::vector<int>& doc_lens;
    const std::vector<double>& type_widths;
    const std::vector<int>& keep_types; 
    const double window_size;
    const std::vector<double>& weights_vec;
    const int weights_mode;
    const bool include_target;
    const std::string decay_type;
    const double decay_param;
    const bool asymmetric;
    const double forward_weight;
    const double backward_weight;
    
    // Output: Reference to thread-local storage
    tbb::enumerable_thread_specific<MapType>& thread_counts;
    
    // Constructor
    FCMBody(const std::vector<const int*>& doc_ptrs,
              const std::vector<int>& doc_lens,
              const std::vector<double>& type_widths,
              const std::vector<int>& keep_types,
              double window_size,
              const std::vector<double>& weights_vec,
              int weights_mode,
              bool include_target,
              std::string decay_type,
              double decay_param,
              bool asymmetric,
              double forward_weight,
              double backward_weight,
              tbb::enumerable_thread_specific<MapType>& thread_counts)
        : doc_ptrs(doc_ptrs), doc_lens(doc_lens), type_widths(type_widths), keep_types(keep_types),
          window_size(window_size), weights_vec(weights_vec), weights_mode(weights_mode),
          include_target(include_target), decay_type(decay_type), decay_param(decay_param),
          asymmetric(asymmetric), forward_weight(forward_weight), backward_weight(backward_weight),
          thread_counts(thread_counts) {}
    
    double calculate_weight(double dist, double window, const std::string& type, double param) const {
        if (dist > window) return 0.0;
        
        if (type == "linear") {
            return std::max(0.0, (window - dist + 1.0) / window); 
        } else if (type == "harmonic") {
            return 1.0 / dist;
        } else if (type == "exponential") {
            return std::exp(-param * dist);
        } else if (type == "power") {
            return std::pow(dist, -param);
        } else { // none
            return 1.0;
        }
    }

    void operator()(const tbb::blocked_range<size_t>& range) const {
        int n_types = type_widths.size();
        bool use_weights_vec = weights_mode > 0;
        int win_int = (int)window_size;
        
        // Get local map
        MapType& counts = thread_counts.local();
        
        for (std::size_t d = range.begin(); d != range.end(); ++d) {
            const int* tokens = doc_ptrs[d];
            int n_tokens = doc_lens[d];
            
            if (n_tokens == 0) continue;
            
            for (int i = 0; i < n_tokens; ++i) {
                int target = tokens[i];
                if (target <= 0 || target > n_types) continue; 
                
                if (!keep_types[target - 1]) continue;

                // Self (Target)
                if (include_target) {
                    double w = 0.0;
                    if (use_weights_vec) {
                        if (weights_mode == 2) { // 0..W
                            if (weights_vec.size() > 0) w = weights_vec[0];
                        } else if (weights_mode == 4) { // -W..W
                            if (win_int >= 0 && win_int < (int)weights_vec.size())
                                w = weights_vec[win_int];
                        }
                    } else {
                        if (decay_type == "harmonic" || decay_type == "power") {
                             w = 1.0; 
                        } else if (decay_type == "linear") {
                             w = (window_size + 1.0) / window_size;
                        } else {
                             w = calculate_weight(0.0, window_size, decay_type, decay_param);
                        }
                    }
                    
                    if (w > 0) counts[make_key(target, target)] += w;
                }

                // Backward context
                double dist = 0;
                for (int j = i - 1; j >= 0; --j) {
                    int context = tokens[j];
                    
                    if (j == i - 1) {
                        dist = 1.0;
                    } else {
                        int intervening_token = tokens[j + 1];
                        if (intervening_token > 0 && intervening_token <= n_types) {
                             dist += type_widths[intervening_token - 1];
                        } else {
                             dist += 1.0; 
                        }
                    }
                    
                    if (dist > window_size) break;
                    
                    if (context <= 0 || context > n_types) continue;
                    if (!keep_types[context - 1]) continue;
                    
                    double w = 0.0;
                    if (use_weights_vec) {
                        if (weights_mode == 1) { // 1..W
                            int dist_idx = i - j;
                            if (dist_idx > 0 && dist_idx <= (int)weights_vec.size()) w = weights_vec[dist_idx - 1];
                        } else if (weights_mode == 2) { // 0..W
                            int dist_idx = i - j;
                            if (dist_idx >= 0 && dist_idx < (int)weights_vec.size()) w = weights_vec[dist_idx]; 
                        } else if (weights_mode == 3) { // -W..-1, 1..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < (int)weights_vec.size()) w = weights_vec[idx];
                        } else if (weights_mode == 4) { // -W..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < (int)weights_vec.size()) w = weights_vec[idx];
                        }
                    } else {
                        w = calculate_weight(dist, window_size, decay_type, decay_param);
                    }
                    
                    w *= backward_weight;
                    
                    if (w > 0) {
                        counts[make_key(target, context)] += w;
                    }
                }
                
                // Forward context
                dist = 0;
                for (int j = i + 1; j < n_tokens; ++j) {
                    int context = tokens[j];
                    
                    if (j == i + 1) {
                        dist = 1.0;
                    } else {
                        int intervening_token = tokens[j - 1];
                        if (intervening_token > 0 && intervening_token <= n_types) {
                            dist += type_widths[intervening_token - 1];
                        } else {
                            dist += 1.0;
                        }
                    }
                    
                    if (dist > window_size) break;
                    
                    if (context <= 0 || context > n_types) continue;
                    if (!keep_types[context - 1]) continue;
                    
                    double w = 0.0;
                    if (use_weights_vec) {
                        if (weights_mode == 1) { // 1..W
                            int dist_idx = j - i;
                            if (dist_idx > 0 && dist_idx <= (int)weights_vec.size()) w = weights_vec[dist_idx - 1];
                        } else if (weights_mode == 2) { // 0..W
                            int dist_idx = j - i;
                            if (dist_idx >= 0 && dist_idx < (int)weights_vec.size()) w = weights_vec[dist_idx];
                        } else if (weights_mode == 3) { // -W..-1, 1..W
                            int idx = win_int + (j - i) - 1;
                            if (idx >= 0 && idx < (int)weights_vec.size()) w = weights_vec[idx];
                        } else if (weights_mode == 4) { // -W..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < (int)weights_vec.size()) w = weights_vec[idx];
                        }
                    } else {
                        w = calculate_weight(dist, window_size, decay_type, decay_param);
                    }
                    
                    w *= forward_weight;
                    
                    if (w > 0) {
                        counts[make_key(target, context)] += w;
                    }
                }
            }
        }
    }
};

// [[Rcpp::export]]
List fcm_cpp(List tokens_list, 
             NumericVector type_widths_r,
             LogicalVector keep_types_r,
             double window_size,
             NumericVector weights_vec_r,
             int weights_mode,
             bool include_target,
             std::string decay_type,
             double decay_param,
             bool asymmetric,
             double forward_weight,
             double backward_weight,
             bool verbose,
             int n_threads = -1) {
    
    // Prepare data for parallel execution
    int n_docs = tokens_list.size();
    std::vector<const int*> doc_ptrs(n_docs);
    std::vector<int> doc_lens(n_docs);
    
    for (int i = 0; i < n_docs; ++i) {
        SEXP doc_sexp = tokens_list[i];
        if (TYPEOF(doc_sexp) == INTSXP) {
            IntegerVector doc = doc_sexp;
            doc_ptrs[i] = doc.begin(); 
            doc_lens[i] = doc.size();
        } else {
            doc_ptrs[i] = nullptr;
            doc_lens[i] = 0;
        }
    }
    
    std::vector<double> type_widths = Rcpp::as<std::vector<double>>(type_widths_r);
    std::vector<int> keep_types(keep_types_r.size());
    for(int i=0; i<keep_types_r.size(); ++i) keep_types[i] = keep_types_r[i];
    
    std::vector<double> weights_vec;
    if (weights_mode > 0) {
        weights_vec = Rcpp::as<std::vector<double>>(weights_vec_r);
    }
    
    // Thread-local storage
    tbb::enumerable_thread_specific<MapType> thread_counts;
    
    FCMBody body(doc_ptrs, doc_lens, type_widths, keep_types, window_size, weights_vec, weights_mode,
                     include_target, decay_type, decay_param, asymmetric, forward_weight, backward_weight,
                     thread_counts);
    
    if (verbose) Rcout << "  Processing " << n_docs << " documents..." << std::endl;

    if (n_threads > 0) {
        tbb::task_arena arena(n_threads);
        arena.execute([&]{
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_docs), body);
        });
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_docs), body);
    }
    
    // Merge results
    if (verbose) Rcout << "  Merging thread-local results..." << std::endl;
    MapType merged_counts;
    for (auto const& local_map : thread_counts) {
        for (auto const& [key, val] : local_map) {
            merged_counts[key] += val;
        }
    }
    
    // Export
    if (verbose) Rcout << "  Constructing sparse matrix triplets..." << std::endl;
    R_xlen_t n = merged_counts.size();
    IntegerVector i(n);
    IntegerVector j(n);
    NumericVector x(n);
    
    R_xlen_t idx = 0;
    for (auto const& [key, val] : merged_counts) {
        std::pair<unsigned int, unsigned int> pair = split_key(key);
        i[idx] = pair.first - 1; 
        j[idx] = pair.second - 1; 
        x[idx] = val;
        idx++;
    }
    
    return List::create(
        Named("i") = i,
        Named("j") = j,
        Named("x") = x,
        Named("dims") = IntegerVector::create(type_widths.size(), type_widths.size())
    );
}
