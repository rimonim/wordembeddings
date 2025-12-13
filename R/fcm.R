#' Create a Feature Co-occurrence Matrix
#'
#' @param x A quanteda `tokens` object.
#' @param context Either a [context_spec] object or parameters passed to create one.
#'   If provided as a `context_spec`, other context-related parameters are ignored.
#' @param window Size of the context window (in words) on either side of the target word.
#'   Ignored if `context` is a `context_spec` object.
#' @param weights Either a string specifying a decay function of distance ("linear", 
#'   "harmonic", "exponential", "power", or "none") or a numeric vector of weights. 
#'   Ignored if `context` is a `context_spec` object.
#' @param weights_args List of arguments for the decay function (e.g. `alpha` for power/exp).
#'   Ignored if `context` is a `context_spec` object.
#' @param distance_metric Metric to input to decay function: "words", "characters", 
#'   "surprisal". Ignored if `context` is a `context_spec` object.
#' @param direction String ("symmetric", "forward", "backward") or numeric ratio of 
#'   forward to backward weight. Ignored if `context` is a `context_spec` object.
#' @param include_target Logical. If TRUE, the target word is included in the context 
#'   (distance 0). Ignored if `context` is a `context_spec` object.
#' @param tri Logical. If TRUE, return only the upper triangle (if symmetric).
#' @param vocab_size Optional. Limit vocabulary to top N most frequent types.
#' @param vocab_coverage Optional. Limit vocabulary to cover this proportion of tokens.
#' @param vocab_keep Optional character vector of types to keep.
#' @param verbose Logical.
#' @param threads Integer. Number of threads to use for parallel processing. If NULL (default), uses all available cores.
#' @export
fcm <- function(x, 
                context = NULL,
                window = 5L,
                weights = "linear",
                weights_args = list(),
                distance_metric = c("words", "characters", "surprisal"),
                direction = "symmetric",
                include_target = FALSE,
                tri = FALSE,
                vocab_size = NULL,
                vocab_coverage = NULL,
                vocab_keep = NULL,
                verbose = FALSE,
                threads = NULL) {
    
    # Handle context_spec object
    if (inherits(context, "context_spec")) {
      window <- context$window
      weights <- context$weights
      weights_args <- context$weights_args
      distance_metric <- context$distance_metric
      direction <- context$direction
      include_target <- context$include_target
    } else if (!is.null(context)) {
      stop("context must be a context_spec object or NULL")
    }
    
    if (!is.character(distance_metric)) {
      # Already set from context_spec
    } else {
      distance_metric <- match.arg(distance_metric)
    }
    
    if (!inherits(x, "tokens")) stop("x must be a quanteda tokens object")
    
    # Handle threads
    n_threads <- -1L
    if (!is.null(threads)) {
        n_threads <- as.integer(threads)
        if (n_threads < 1) stop("threads must be a positive integer")
    }
    
    if (verbose) {
        cat("Constructing FCM...\n")
        cat(sprintf("  Window size: %d\n", window))
        cat(sprintf("  Weighting: %s\n", if(is.character(weights)) weights else "custom"))
        cat(sprintf("  Threads: %s\n", if(n_threads > 0) n_threads else "auto"))
    }
    
    # Handle weights
    weights_vec <- numeric(0)
    weights_mode <- 0L # 0: decay, 1: 1..W, 2: 0..W, 3: -W..-1,1..W, 4: -W..W
    decay_type <- "none"
    
    if (is.numeric(weights)) {
        L <- length(weights)
        if (include_target) {
             if (L == window + 1) {
                 weights_mode <- 2L
             } else if (L == 2 * window + 1) {
                 weights_mode <- 4L
             } else {
                 stop("Length of weights vector must be window+1 or 2*window+1 when include_target is TRUE")
             }
        } else {
             if (L == window) {
                 weights_mode <- 1L
             } else if (L == 2 * window) {
                 weights_mode <- 3L
             } else {
                 stop("Length of weights vector must be window or 2*window when include_target is FALSE")
             }
        }
        weights_vec <- weights
        if (distance_metric != "words") {
            warning("Custom weight vector provided; ignoring distance_metric and using token distance.")
            distance_metric <- "words"
        }
    } else if (is.character(weights)) {
        decay_type <- match.arg(weights, c("linear", "harmonic", "exponential", "power", "none"))
    } else {
        stop("weights must be a numeric vector or a decay function name")
    }
    
    # Handle direction
    forward_weight <- 1.0
    backward_weight <- 1.0
    ordered <- FALSE
    
    if (is.numeric(direction)) {
        if (direction == 1) {
            ordered <- FALSE
        } else {
            forward_weight <- direction
            ordered <- TRUE
        }
    } else {
        if (direction == "symmetric") {
            ordered <- FALSE
        } else if (direction == "forward") {
            backward_weight <- 0.0
            ordered <- TRUE
        } else if (direction == "backward") {
            forward_weight <- 0.0
            ordered <- TRUE
        } else {
            stop('direction must be "symmetric", "forward", "backward", or a numeric ratio')
        }
    }
    
    types <- quanteda::types(x)
    n_types <- length(types)
    
    # 1. Determine Vocabulary to Keep
    keep_types <- rep(TRUE, n_types)
    
    if (!is.null(vocab_keep) || !is.null(vocab_size) || !is.null(vocab_coverage)) {
        if (!is.null(vocab_size) || !is.null(vocab_coverage)) {
             freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
             
             if (!is.null(vocab_size)) {
                 limit_types <- head(names(freqs), vocab_size)
                 keep_types <- keep_types & (types %in% limit_types)
             }
             
             if (!is.null(vocab_coverage)) {
                 total_tokens <- sum(freqs)
                 cum_freq <- cumsum(freqs) / total_tokens
                 cutoff_idx <- which(cum_freq >= vocab_coverage)[1]
                 if (is.na(cutoff_idx)) cutoff_idx <- length(freqs)
                 coverage_types <- head(names(freqs), cutoff_idx)
                 keep_types <- keep_types & (types %in% coverage_types)
             }
        }
        
        if (!is.null(vocab_keep)) {
            keep_types <- keep_types & (types %in% vocab_keep)
        }
    }
    
    # 2. Determine Widths
    type_widths <- rep(1.0, n_types)
    if (distance_metric == "characters") {
        freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
        m <- match(types, names(freqs))
        type_widths <- nchar(types)
        # set average width to 1
        type_widths <- type_widths/weighted.mean(type_widths,w=freqs[m])
    } else if (distance_metric == "surprisal") {
        # Weight = -log(probability)
        freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
        total_tokens <- sum(freqs)
        
        m <- match(types, names(freqs))
        probs <- freqs[m] / total_tokens
        probs[is.na(probs) | probs == 0] <- 1.0 / total_tokens 
        type_widths <- -log(probs)
        # set average width to 1
        type_widths <- type_widths/weighted.mean(type_widths,w=probs)
    }
    
    # 3. Call C++
    # Use unclass() to get the underlying list of integer vectors
    tokens_list <- unclass(x)
    
    decay_param <- if (!is.null(weights_args$alpha)) weights_args$alpha else 1.0
    
    result <- fcm_cpp(tokens_list, type_widths, keep_types, window, weights_vec, weights_mode, include_target,
                      decay_type, decay_param, 
                      ordered, 
                      forward_weight, backward_weight, verbose, n_threads)
    
    # 4. Construct Matrix
    mat <- Matrix::sparseMatrix(i = result$i, j = result$j, x = result$x, 
                                dims = result$dims, index1 = FALSE,
                                dimnames = list(types, types))
    
    # 5. Post-process
    if (any(!keep_types)) {
        mat <- mat[keep_types, keep_types]
    }
    
    if (!ordered) {
        # Symmetrize if not ordered
        if (tri) {
            mat <- Matrix::triu(mat)
        }
    }
    
    fcm <- quanteda::as.fcm(mat)
    fcm@meta$object$count <- "weighted"
    fcm
}
