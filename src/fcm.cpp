#include <Rcpp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Helper to combine two 32-bit ints into one 64-bit key
inline unsigned long long make_key(unsigned int row, unsigned int col) {
    return ((unsigned long long)row << 32) | col;
}

inline std::pair<unsigned int, unsigned int> split_key(unsigned long long key) {
    return std::make_pair((unsigned int)(key >> 32), (unsigned int)(key & 0xFFFFFFFF));
}

typedef std::unordered_map<unsigned long long, double> MapType;

// Helper function to calculate weight
inline double calculate_weight(double dist, double window, const std::string& type, double param) {
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

// Process a single document and accumulate counts into local_counts
inline void process_document(
    const int* tokens,
    int n_tokens,
    const std::vector<double>& type_widths,
    const std::vector<int>& keep_types,
    double window_size,
    const std::vector<double>& weights_vec,
    int weights_mode,
    bool include_target,
    const std::string& decay_type,
    double decay_param,
    double forward_weight,
    double backward_weight,
    MapType& counts
) {
    int n_types = type_widths.size();
    bool use_weights_vec = weights_mode > 0;
    int win_int = (int)window_size;
    
    if (n_tokens == 0) return;
    
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
    
    // Determine number of threads
    int actual_threads = n_threads;
#ifdef _OPENMP
    if (actual_threads <= 0) actual_threads = omp_get_max_threads();
#else
    actual_threads = 1;
#endif
    
    if (verbose) Rcout << "  Processing " << n_docs << " documents with " << actual_threads << " threads..." << std::endl;
    
    // Thread-local storage for OpenMP
    std::vector<MapType> thread_maps(actual_threads);
    
#ifdef _OPENMP
#pragma omp parallel num_threads(actual_threads)
    {
        int tid = omp_get_thread_num();
        MapType& local_counts = thread_maps[tid];
        
#pragma omp for schedule(dynamic)
        for (int d = 0; d < n_docs; ++d) {
            process_document(
                doc_ptrs[d], doc_lens[d],
                type_widths, keep_types, window_size,
                weights_vec, weights_mode, include_target,
                decay_type, decay_param,
                forward_weight, backward_weight,
                local_counts
            );
        }
    }
#else
    // Single-threaded fallback
    MapType& local_counts = thread_maps[0];
    for (int d = 0; d < n_docs; ++d) {
        process_document(
            doc_ptrs[d], doc_lens[d],
            type_widths, keep_types, window_size,
            weights_vec, weights_mode, include_target,
            decay_type, decay_param,
            forward_weight, backward_weight,
            local_counts
        );
    }
#endif
    
    // Merge results
    if (verbose) Rcout << "  Merging thread-local results..." << std::endl;
    MapType merged_counts;
    for (const auto& local_map : thread_maps) {
        for (const auto& [key, val] : local_map) {
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
