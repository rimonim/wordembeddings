#include <Rcpp.h>
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

class FCMAccumulator {
public:
    std::unordered_map<unsigned long long, double> counts;
    
    FCMAccumulator() {}
    
    double calculate_weight(double dist, double window, const std::string& type, double param) {
        if (dist > window) return 0.0;
        
        if (type == "linear") {
            // Linear decay: 1 at dist=1, 0 at dist > window
            // Formula: (window - dist + 1) / window
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

    void process_documents(List tokens_list, 
                          NumericVector type_widths,
                          LogicalVector keep_types,
                          double window_size,
                          NumericVector weights_vec_r,
                          int weights_mode, // 0: decay, 1: 1..W, 2: 0..W, 3: -W..-1,1..W, 4: -W..W
                          bool include_target,
                          std::string decay_type,
                          double decay_param,
                          bool asymmetric,
                          double forward_weight,
                          double backward_weight,
                          bool verbose) {
        
        int n_docs = tokens_list.size();
        int n_types = type_widths.size();
        bool use_weights_vec = weights_mode > 0;
        int win_int = (int)window_size;
        
        // Convert Rcpp vector to std::vector for safer/faster access
        std::vector<double> weights_vec;
        if (use_weights_vec) {
            weights_vec = Rcpp::as<std::vector<double>>(weights_vec_r);
        }
        
        for (int d = 0; d < n_docs; ++d) {
            if (verbose && d % 1000 == 0) Rcpp::checkUserInterrupt();
            
            SEXP doc_sexp = tokens_list[d];
            IntegerVector tokens;
            
            if (TYPEOF(doc_sexp) == INTSXP) {
                tokens = doc_sexp;
            } else if (TYPEOF(doc_sexp) == REALSXP) {
                // Handle numeric vectors (e.g. from manual list creation) by coercing to integer
                tokens = Rcpp::as<IntegerVector>(doc_sexp);
            } else {
                // Skip other types (NULL, character, etc.)
                continue;
            }

            int n_tokens = tokens.size();
            if (n_tokens == 0) continue;
            
            for (int i = 0; i < n_tokens; ++i) {
                int target = tokens[i];
                // quanteda is 1-based. 0 might be padding.
                if (target <= 0 || target > n_types) continue; 
                
                // If target is not in our vocab of interest, we skip it as a ROW
                if (!keep_types[target - 1]) continue;

                // Self (Target)
                if (include_target) {
                    double w = 0.0;
                    if (use_weights_vec) {
                        if (weights_mode == 2) { // 0..W
                            if (weights_vec.size() > 0) w = weights_vec[0];
                        } else if (weights_mode == 4) { // -W..W
                            if (win_int >= 0 && win_int < weights_vec.size())
                                w = weights_vec[win_int];
                        }
                    } else {
                        // Decay function at dist 0
                        // Default to 1.0 for singularities
                        if (decay_type == "harmonic" || decay_type == "power") {
                             w = 1.0; 
                        } else if (decay_type == "linear") {
                             // (W - 0 + 1) / W = 1 + 1/W
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
                    
                    // Distance calculation
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
                            if (dist_idx > 0 && dist_idx <= weights_vec.size()) w = weights_vec[dist_idx - 1];
                        } else if (weights_mode == 2) { // 0..W
                            int dist_idx = i - j;
                            if (dist_idx >= 0 && dist_idx < weights_vec.size()) w = weights_vec[dist_idx]; 
                        } else if (weights_mode == 3) { // -W..-1, 1..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < weights_vec.size()) w = weights_vec[idx];
                        } else if (weights_mode == 4) { // -W..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < weights_vec.size()) w = weights_vec[idx];
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
                            if (dist_idx > 0 && dist_idx <= weights_vec.size()) w = weights_vec[dist_idx - 1];
                        } else if (weights_mode == 2) { // 0..W
                            int dist_idx = j - i;
                            if (dist_idx >= 0 && dist_idx < weights_vec.size()) w = weights_vec[dist_idx];
                        } else if (weights_mode == 3) { // -W..-1, 1..W
                            int idx = win_int + (j - i) - 1;
                            if (idx >= 0 && idx < weights_vec.size()) w = weights_vec[idx];
                        } else if (weights_mode == 4) { // -W..W
                            int idx = (j - i) + win_int;
                            if (idx >= 0 && idx < weights_vec.size()) w = weights_vec[idx];
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
    
    List export_matrix(int n_types) {
        R_xlen_t n = counts.size();
        IntegerVector i(n);
        IntegerVector j(n);
        NumericVector x(n);
        
        R_xlen_t idx = 0;
        for (auto const& [key, val] : counts) {
            std::pair<unsigned int, unsigned int> pair = split_key(key);
            i[idx] = pair.first - 1; // 0-based for sparseMatrix
            j[idx] = pair.second - 1; // 0-based for sparseMatrix
            x[idx] = val;
            idx++;
        }
        
        return List::create(
            Named("i") = i,
            Named("j") = j,
            Named("x") = x,
            Named("dims") = IntegerVector::create(n_types, n_types)
        );
    }
};

// [[Rcpp::export]]
List fcm_cpp(List tokens_list, 
             NumericVector type_widths,
             LogicalVector keep_types,
             double window_size,
             NumericVector weights_vec,
             int weights_mode,
             bool include_target,
             std::string decay_type,
             double decay_param,
             bool asymmetric,
             double forward_weight,
             double backward_weight,
             bool verbose) {
    
    FCMAccumulator acc;
    acc.process_documents(tokens_list, type_widths, keep_types, window_size, weights_vec, weights_mode, include_target,
                         decay_type, decay_param, asymmetric, forward_weight, backward_weight, verbose);
    
    return acc.export_matrix(type_widths.size());
}
