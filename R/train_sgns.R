#' Train embeddings with Skip-Gram with Negative Sampling
#'
#' Train word and/or context embeddings using the Skip-Gram with Negative 
#' Sampling (SGNS) algorithm (Mikolov et al., 2013). Supports both streaming from 
#' tokens and training from pre-computed FCMs, though the latter is generally much slower.
#'
#' @param x a quanteda [`tokens`][quanteda::tokens] object or feature co-occurrence matrix (FCM).
#' @param context A [context_spec] object defining the context window configuration
#'   and vocabulary parameters. Required for `train_sgns.tokens`.
#' @param n_dims integer. Dimensionality of embeddings.
#' @param neg integer. Number of negative samples per positive example. Default is 5.
#' @param lr numeric. Initial learning rate. Default is 0.05.
#' @param epochs integer. Number of passes through the data. Default is 5.
#' @param init character. Initialization: "uniform" or "normal". Default is "uniform".
#' @param seed integer. Random seed for reproducibility.
#' @param verbose logical. Print progress information.
#' @param threads integer. Number of threads. Default uses all available cores.
#' @param ... Additional arguments passed to methods.
#'
#' @references
#' Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). 
#' Distributed Representations of Words and Phrases and their Compositionality. 
#' In C. J. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q. Weinberger (Eds.), 
#' Advances in Neural Information Processing Systems (Vol. 26). Curran Associates, Inc.
#'
#' @export
train_sgns <- function(x, ...) {
  UseMethod("train_sgns")
}

#' @rdname train_sgns
#' @method train_sgns tokens
#' @export
train_sgns.tokens <- function(
  x,
  context = context_spec(),
  n_dims = 100,
  neg = 5,
  lr = 0.05,
  epochs = 5,
  init = "uniform",
  seed = NULL,
  verbose = TRUE,
  threads = parallel::detectCores(),
  ...
) {
  
  if (!inherits(context, "context_spec")) {
    stop("context must be a context_spec object. Use context_spec() to create one.")
  }
  
  if (!inherits(x, "tokens")) {
    stop("x must be a quanteda tokens object")
  }
  
  if (is.null(seed)) seed <- 1L
  
  # Extract all parameters from context_spec
  window_size <- context$window
  min_count <- context$min_count
  vocab_size <- context$vocab_size
  vocab_coverage <- context$vocab_coverage
  vocab_keep <- context$vocab_keep
  context_smoothing <- context$context_smoothing
  subsample <- context$subsample
  weights <- context$weights
  weights_args <- context$weights_args
  distance_metric <- context$distance_metric
  direction <- context$direction
  include_target <- context$include_target
  
  # Apply vocabulary filtering in R (matching fcm() logic exactly)
  types <- quanteda::types(x)
  n_types <- length(types)
  keep_types <- rep(TRUE, n_types)
  
  # Get frequencies once (used by multiple filters)
  freqs <- NULL
  if (!is.null(vocab_keep) || !is.null(vocab_size) || !is.null(vocab_coverage) || !is.null(min_count)) {
    freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
    # Sort by frequency (descending) for vocab_size and vocab_coverage
    freqs <- sort(freqs, decreasing = TRUE)
  }
  
  # Apply vocab filters in order: vocab_size → vocab_coverage → min_count → vocab_keep
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
  
  if (!is.null(min_count) && min_count > 1) {
    # Match types with their frequencies
    m <- match(types, names(freqs))
    type_freqs <- freqs[m]
    type_freqs[is.na(type_freqs)] <- 0
    
    # Apply min_count filter
    min_count_filter <- type_freqs >= min_count
    keep_types <- keep_types & min_count_filter
  }
  
  if (!is.null(vocab_keep)) {
    # Force include vocab_keep words (OR condition)
    keep_types <- keep_types | (types %in% vocab_keep)
  }

  # Get list of types to keep - will pass to C++ to filter vocabulary
  # This ensures C++ uses exactly the same vocabulary as fcm()
  types_to_keep <- types[keep_types]
  
  # Recompute types and widths for the original (unfiltered) tokens
  # C++ will handle the actual filtering based on types_to_keep
  types <- quanteda::types(x)
  n_types <- length(types)
  type_widths <- rep(1.0, n_types)
  
  if (distance_metric == "characters") {
    freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
    m <- match(types, names(freqs))
    type_widths <- nchar(types)
    type_widths <- type_widths / weighted.mean(type_widths, w = freqs[m])
  } else if (distance_metric == "surprisal") {
    freqs <- colSums(quanteda::dfm(x, tolower = FALSE))
    total_tokens <- sum(freqs)
    m <- match(types, names(freqs))
    probs <- freqs[m] / total_tokens
    probs[is.na(probs) | probs == 0] <- 1.0 / total_tokens
    type_widths <- -log(probs)
    type_widths <- type_widths / weighted.mean(type_widths, w = probs)
  }
  
  # Handle direction parameter
  forward_weight <- 1.0
  backward_weight <- 1.0
  
  if (is.numeric(direction)) {
    if (direction != 1) {
      forward_weight <- direction
    }
  } else {
    if (direction == "forward") {
      backward_weight <- 0.0
    } else if (direction == "backward") {
      forward_weight <- 0.0
    }
  }
  
  # Handle weights parameter
  weights_vec <- numeric(0)
  weights_mode <- 0L
  decay_type <- "none"
  decay_alpha <- 1.0
  
  if (is.numeric(weights)) {
    weights_vec <- weights
    L <- length(weights)
    if (include_target) {
      if (L == window_size + 1) {
        weights_mode <- 2L
      } else if (L == 2 * window_size + 1) {
        weights_mode <- 4L
      } else {
        stop("Length of weights vector must be window+1 or 2*window+1 when include_target is TRUE")
      }
    } else {
      if (L == window_size) {
        weights_mode <- 1L
      } else if (L == 2 * window_size) {
        weights_mode <- 3L
      } else {
        stop("Length of weights vector must be window or 2*window when include_target is FALSE")
      }
    }
  } else if (is.character(weights)) {
    decay_type <- match.arg(weights, c("linear", "harmonic", "exponential", "power", "none"))
    if (!is.null(weights_args$alpha)) {
      decay_alpha <- weights_args$alpha
    }
  }
  
  # Pass the filtered vocabulary as integer indices to C++
  # Match types_to_keep to types to get their indices (1-indexed)
  vocab_indices <- match(types_to_keep, types)
  
  # Call enhanced streaming C++ implementation
  # Note: vocabulary filtering done in R, passed as integer indices
  result <- sgns_streaming_cpp(
    tokens_list = x,
    vocab = as.integer(vocab_indices),  # 1-indexed type indices to include
    type_widths = type_widths,
    n_dims = as.integer(n_dims),
    n_neg = as.integer(neg),
    window = as.integer(window_size),
    lr = lr,
    epochs = as.integer(epochs),
    context_smoothing = context_smoothing,
    subsample = subsample,
    weights_type = decay_type,
    weights_alpha = decay_alpha,
    weights_vec = weights_vec,
    weights_mode = as.integer(weights_mode),
    include_target = include_target,
    forward_weight = forward_weight,
    backward_weight = backward_weight,
    init_type = init,
    seed = as.integer(seed),
    verbose = verbose,
    threads = as.integer(threads)
  )
  
  # Add vocabulary as row names using the order returned from C++
  # C++ sorts by frequency, so we need to match that order
  vocab_order <- result$vocab_indices  # 1-indexed type indices in frequency order
  vocab_words <- types[vocab_order]
  rownames(result$word_embeddings) <- vocab_words
  rownames(result$context_embeddings) <- vocab_words
  
  result
}

#' @rdname train_sgns
#' @method train_sgns Matrix
#' @export
train_sgns.Matrix <- function(
  x,
  n_dims = 100,
  neg = 5,
  lr = 0.05,
  epochs = 5,
  init = "uniform",
  seed = NULL,
  verbose = TRUE,
  threads = parallel::detectCores(),
  ...
) {
  # Dispatch to fcm method
  train_sgns.fcm(x, n_dims, neg, lr, epochs, init, seed, verbose, threads, ...)
}

#' @rdname train_sgns
#' @method train_sgns fcm
#' @export
train_sgns.fcm <- function(
  x,
  n_dims = 100,
  neg = 5,
  lr = 0.05,
  epochs = 5,
  init = "uniform",
  seed = NULL,
  verbose = TRUE,
  threads = parallel::detectCores(),
  ...
) {
  
  fcm <- x  # Rename for clarity
  
  # Input validation
  stopifnot(
    "`fcm` must be a Quanteda fcm, sparseMatrix, SparseArray, or array" =
      (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") ||
       inherits(fcm, "SparseArray") || is.array(fcm)) &&
      (length(dim(fcm)) %in% c(2, 3)),
    "`n_dims` must be a positive integer" = is.numeric(n_dims) && n_dims > 0,
    "`neg` must be a positive integer" = is.numeric(neg) && neg > 0,
    "`lr` must be a positive number" = is.numeric(lr) && lr > 0,
    "`epochs` must be a positive integer" = is.numeric(epochs) && epochs > 0,
    "`init` must be 'uniform' or 'normal'" = init %in% c("uniform", "normal"),
    "`verbose` must be logical" = is.logical(verbose),
    "`threads` must be a positive integer" = is.numeric(threads) && threads > 0
  )

  if (is.null(seed)) seed <- 1L
  
  n_dims <- as.integer(n_dims)
  neg <- as.integer(neg)
  epochs <- as.integer(epochs)
  threads <- as.integer(threads)

  # Handle 3D arrays
  if (length(dim(fcm)) == 3) {
    return(.train_sgns_3d(
      fcm, n_dims, neg, lr, epochs, init, seed, verbose, threads
    ))
  }

  # Convert to sparse matrix format for consistent handling
  if (inherits(fcm, "fcm")) {
    fcm_sparse <- methods::as(fcm, "TsparseMatrix")
  } else if (inherits(fcm, "sparseMatrix")) {
    fcm_sparse <- methods::as(fcm, "TsparseMatrix")
  } else if (inherits(fcm, "SparseArray")) {
    fcm_sparse <- as(fcm, "TsparseMatrix")
  } else {
    fcm_sparse <- methods::as(as.matrix(fcm), "TsparseMatrix")
  }

  # Get vocabulary
  vocab <- rownames(fcm_sparse)
  if (is.null(vocab)) {
    vocab <- paste0("word_", seq_len(nrow(fcm_sparse)))
  }
  
  # Call C++ implementation that samples from FCM like streaming
  result <- sgns_from_fcm_cpp(
    i_indices = fcm_sparse@i,
    j_indices = fcm_sparse@j,
    x_values = fcm_sparse@x,
    n_words = nrow(fcm_sparse),
    n_contexts = ncol(fcm_sparse),
    n_dims = n_dims,
    n_neg = neg,
    lr = lr,
    epochs = epochs,
    init_type = init,
    seed = as.integer(seed),
    verbose = verbose,
    threads = threads
  )
  
  # Add vocabulary as row names
  rownames(result$word_embeddings) <- vocab
  rownames(result$context_embeddings) <- vocab
  
  result
}

#' @keywords internal
#' Handle 3D FCM arrays
.train_sgns_3d <- function(fcm, n_dims, neg, lr, epochs, init, seed, verbose, threads) {
  fcm_ids <- dimnames(fcm)[[3]]
  fcm_list <- lapply(seq_len(dim(fcm)[3]), function(i) {
    train_sgns.fcm(
      fcm[, , i],
      n_dims = n_dims,
      neg = neg,
      lr = lr,
      epochs = epochs,
      init = init,
      seed = seed,
      verbose = verbose,
      threads = threads
    )
  })

  # Combine embeddings into 3D arrays
  word_emb_list <- lapply(fcm_list, `[[`, "word_embeddings")
  word_embeddings <- S4Arrays::abind(word_emb_list, along = 3)
  
  context_emb_list <- lapply(fcm_list, `[[`, "context_embeddings")
  context_embeddings <- S4Arrays::abind(context_emb_list, along = 3)

  # Preserve dimension names
  if (!is.null(dimnames(fcm)[[3]])) {
    dimnames(word_embeddings)[[3]] <- fcm_ids
    dimnames(context_embeddings)[[3]] <- fcm_ids
  }

  list(
    word_embeddings = word_embeddings,
    context_embeddings = context_embeddings
  )
}
