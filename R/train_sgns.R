#' Train embeddings with Skip-Gram with Negative Sampling
#'
#' Generic function for training word embeddings using the Skip-Gram with Negative 
#' Sampling (SGNS) algorithm (Mikolov et al., 2013). Supports both streaming from 
#' tokens (fast, like word2vec) and training from pre-computed FCMs (consistent 
#' with other methods in this package).
#'
#' @param x Either a quanteda `tokens` object or a feature co-occurrence matrix (FCM).
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
#' @param context A [context_spec] object defining the context window configuration.
#'   Required for `train_sgns.tokens`.
#' @param vocab_size Optional. Limit vocabulary to top N most frequent types.
#' @param vocab_coverage Optional. Limit vocabulary to cover this proportion of tokens.
#' @param vocab_keep Optional character vector of types to keep.
#' @param min_count integer. Minimum frequency for a word to be included in vocabulary. Default is 5.
#' @param n_dims integer. Dimensionality of embeddings.
#' @param neg integer. Number of negative samples per positive example. Default is 5.
#' @param lr numeric. Initial learning rate. Default is 0.05.
#' @param epochs integer. Number of passes through the data. Default is 5.
#' @param context_smoothing numeric. Power to raise context frequencies for negative 
#'   sampling. Default is 0.75.
#' @param subsample numeric. Subsampling threshold for frequent words. Default is 1e-3.
#' @param init character. Initialization: "uniform" or "normal". Default is "uniform".
#' @param seed integer. Random seed for reproducibility.
#' @param verbose logical. Print progress information.
#' @param threads integer. Number of threads. Default uses all available cores.
train_sgns.tokens <- function(
  x,
  context = context_spec(),
  vocab_size = NULL,
  vocab_coverage = NULL,
  vocab_keep = NULL,
  min_count = 5,
  n_dims = 100,
  neg = 5,
  lr = 0.05,
  epochs = 5,
  context_smoothing = 0.75,
  subsample = 1e-3,
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
  
  # Get window size from context_spec
  window_size <- context$window
  
  # Determine vocab_size (0 means unlimited)
  vocab_limit <- if (is.null(vocab_size)) 0L else as.integer(vocab_size)
  
  # Call streaming C++ implementation
  # Pass tokens object directly (not as.list) to preserve attributes
  result <- sgns_streaming_cpp(
    tokens_list = x,
    min_count = as.integer(min_count),
    vocab_size = vocab_limit,
    n_dims = as.integer(n_dims),
    n_neg = as.integer(neg),
    window = as.integer(window_size),
    lr = lr,
    epochs = as.integer(epochs),
    context_smoothing = context_smoothing,
    subsample = subsample,
    init_type = init,
    seed = as.integer(seed),
    verbose = verbose,
    threads = as.integer(threads)
  )
  
  result
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
  grain_size = 1,
  context_smoothing = 0.75,
  target_smoothing = 1,
  subsample = 0,
  reject_positives = TRUE,
  init = "uniform",
  bootstrap_positive = FALSE,
  output = "word_embeddings",
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
    "`grain_size` must be a positive integer" = is.numeric(grain_size) && grain_size > 0,
    "`context_smoothing` must be a non-negative number" = is.numeric(context_smoothing) && context_smoothing >= 0,
    "`target_smoothing` must be a non-negative number" = is.numeric(target_smoothing) && target_smoothing >= 0,
    "`subsample` must be a non-negative number" = is.numeric(subsample) && subsample >= 0,
    "`reject_positives` must be logical" = is.logical(reject_positives),
    "`init` must be 'uniform' or 'normal'" = init %in% c("uniform", "normal"),
    "`bootstrap_positive` must be logical" = is.logical(bootstrap_positive),
    "`output` must be 'word_embeddings', 'context_embeddings', or 'all'" =
      output %in% c("word_embeddings", "context_embeddings", "all"),
    "`verbose` must be logical" = is.logical(verbose),
    "`threads` must be a positive integer" = is.numeric(threads) && threads > 0
  )

  if (!is.null(seed)) {
    set.seed(seed)
  }

  n_dims <- as.integer(n_dims)
  neg <- as.integer(neg)
  epochs <- as.integer(epochs)
  grain_size <- as.integer(grain_size)
  threads <- as.integer(threads)

  # Handle 3D arrays
  if (length(dim(fcm)) == 3) {
    return(.train_sgns_3d(
      fcm, n_dims, neg, lr, epochs, grain_size,
      context_smoothing, target_smoothing, subsample, reject_positives, init, bootstrap_positive, 
      output, verbose, seed, threads
    ))
  }

  # Extract FCM metadata
  is_quanteda <- inherits(fcm, "fcm")
  if (is_quanteda) {
    fcm_meta <- fcm@meta
  }

  # Convert to sparse matrix format for consistent handling
  if (inherits(fcm, "sparseMatrix") || is_quanteda) {
    fcm_sparse <- methods::as(fcm, "TsparseMatrix")
  } else if (inherits(fcm, "SparseArray")) {
    fcm_sparse <- as(fcm, "TsparseMatrix")
  } else {
    fcm_sparse <- methods::as(as.matrix(fcm), "TsparseMatrix")
  }

  n_words <- nrow(fcm_sparse)
  n_contexts <- ncol(fcm_sparse)
  
  # Apply target smoothing: reweight based on row (word) frequencies
  x_values <- fcm_sparse@x
  if (target_smoothing != 1) {
    row_sums <- Matrix::rowSums(fcm_sparse)
    # Compute row weight: rowsum^target_smoothing
    row_weights <- row_sums^target_smoothing
    # Normalize to preserve total mass
    row_weights <- row_weights / mean(row_weights)
    # Apply to each element
    x_values <- x_values * row_weights[fcm_sparse@i + 1]
  }
  
  # Apply subsampling: downweight frequent pairs
  if (subsample > 0) {
    total_count <- sum(x_values)
    pair_freq <- x_values / total_count
    # Keep probability: min(1, sqrt(t/f) + t/f)
    keep_prob <- pmin(1, sqrt(subsample / pair_freq) + subsample / pair_freq)
    x_values <- x_values * keep_prob
  }

  # Set number of threads
  if (threads < 1) threads <- 1L
  threads <- as.integer(threads)

  # Call C++ implementation
  cpp_result <- sgns_train_cpp(
    i_indices = fcm_sparse@i,
    j_indices = fcm_sparse@j,
    x_values = x_values,
    n_words = n_words,
    n_contexts = n_contexts,
    n_dims = n_dims,
    n_neg = neg,
    lr = lr,
    epochs = epochs,
    grain_size = grain_size,
    smoothing = context_smoothing,
    reject_positives = reject_positives,
    init_type = init,
    bootstrap_positive = bootstrap_positive,
    seed = if (is.null(seed)) 0L else as.integer(seed),
    verbose = verbose,
    threads = threads
  )

  # Extract and name embeddings
  word_embeddings <- cpp_result$word_embeddings
  context_embeddings <- cpp_result$context_embeddings
  loss_history <- cpp_result$loss_history

  # Preserve rownames
  if (!is.null(rownames(fcm_sparse))) {
    rownames(word_embeddings) <- rownames(fcm_sparse)
    rownames(context_embeddings) <- colnames(fcm_sparse)
  }

  # Prepare output based on user request
  word_emb <- if (output %in% c("word_embeddings", "all")) {
    word_embeddings
  } else {
    NULL
  }
  context_emb <- if (output %in% c("context_embeddings", "all")) {
    context_embeddings
  } else {
    NULL
  }

  # Create dynamic_embeddings object
  result <- dynamic_embeddings(
    fcm = fcm,
    context_embeddings = context_emb,
    word_embeddings = word_emb,
    train_method = "sgns"
  )
  
  result$loss_history <- loss_history

  # Add SGNS-specific control information
  result$control <- list(
    method = "sgns",
    n_dims = n_dims,
    neg = neg,
    lr = lr,
    epochs = epochs,
    grain_size = grain_size,
    context_smoothing = context_smoothing,
    target_smoothing = target_smoothing,
    subsample = subsample,
    reject_positives = reject_positives,
    init = init,
    bootstrap_positive = bootstrap_positive,
    threads = threads
  )

  result
}

#' @keywords internal
#' Handle 3D FCM arrays
.train_sgns_3d <- function(fcm, n_dims, neg, lr, epochs, grain_size,
                            context_smoothing, target_smoothing, subsample, reject_positives, init, 
                            bootstrap_positive, output, verbose, seed, threads) {
  fcm_ids <- dimnames(fcm)[[3]]
  fcm_list <- lapply(seq_len(dim(fcm)[3]), function(i) {
    train_sgns.fcm(
      fcm[, , i],
      n_dims = n_dims,
      neg = neg,
      lr = lr,
      epochs = epochs,
      grain_size = grain_size,
      context_smoothing = context_smoothing,
      target_smoothing = target_smoothing,
      subsample = subsample,
      reject_positives = reject_positives,
      init = init,
      bootstrap_positive = bootstrap_positive,
      output = output,
      verbose = verbose,
      seed = seed,
      threads = threads
    )
  })

  # Combine embeddings into 3D arrays
  if (!is.null(fcm_list[[1]]$word_embeddings)) {
    word_emb_list <- lapply(fcm_list, `[[`, "word_embeddings")
    word_embeddings <- S4Arrays::abind(word_emb_list, along = 3)
  } else {
    word_embeddings <- NULL
  }

  if (!is.null(fcm_list[[1]]$context_embeddings)) {
    context_emb_list <- lapply(fcm_list, `[[`, "context_embeddings")
    context_embeddings <- S4Arrays::abind(context_emb_list, along = 3)
  } else {
    context_embeddings <- NULL
  }
  
  loss_history <- lapply(fcm_list, `[[`, "loss_history")

  # Preserve dimension names
  if (!is.null(dimnames(fcm)[[3]])) {
    if (!is.null(word_embeddings)) {
      dimnames(word_embeddings)[[3]] <- fcm_ids
    }
    if (!is.null(context_embeddings)) {
      dimnames(context_embeddings)[[3]] <- fcm_ids
    }
  }

  result <- dynamic_embeddings(
    fcm = fcm,
    context_embeddings = context_embeddings,
    word_embeddings = word_embeddings,
    train_method = "sgns"
  )
  
  result$loss_history <- loss_history
  
  result
}
