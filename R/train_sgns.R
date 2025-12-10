#' Train embeddings with Skip-Gram with Negative Sampling
#'
#' `train_sgns(fcm)` trains word and context embeddings using the Skip-Gram with
#' Negative Sampling (SGNS) algorithm (Mikolov et al., 2013). This is a
#' computationally efficient alternative to full softmax that approximates the
#' skip-gram model by learning from binary classification tasks between
#' observed co-occurrences (positive samples) and randomly sampled words (negative samples).
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D
#'	array-like object
#' @param n_dims integer. Dimensionality of embeddings.
#' @param neg integer. Number of negative samples per positive example. Default is 5.
#'  Values of 5-20 are typical for small datasets, 2-5 for large datasets.
#' @param lr numeric. Initial learning rate. Default is 0.05. Learning rate decays
#'  linearly across iterations.
#' @param iterations integer. Number of passes through the FCM. Default is 5.
#'  More iterations improve quality but increase computation time.
#' @param batch_size integer. Controls the grain size for parallel processing.
#'  Default is 1, which allows the parallel backend to automatically determine
#'  the chunk size. Larger values (e.g. 10000) specify the exact number of
#'  pairs processed per thread task, which can tune performance.
#' @param context_smoothing numeric. Power to raise context (column) frequencies when constructing
#'  the negative sampling table. Default is 0.75 (Mikolov et al., 2013). Set to 0 for
#'  uniform sampling. 
#' @param target_smoothing numeric. Power to raise target word (row) frequencies when
#'  reweighting the FCM for training. Default is 1 (no reweighting). Set to 0 for
#'  uniform target sampling (all words trained equally regardless of frequency).
#'  Values < 1 downsample frequent words; values > 1 oversample frequent words.
#' @param subsample numeric. Subsampling threshold for downweighting frequent pairs,
#'  following Mikolov et al. (2013). Default is 0 (no subsampling). Typical values
#'  are 1e-3 to 1e-5. Each pair is kept with probability
#'  `min(1, sqrt(subsample / freq) + subsample / freq)`.
#' @param reject_positives logical. If `TRUE` (default), rejection sampling prevents
#'  selecting the positive context as a negative sample. If `FALSE`, positive and
#'  negative samples are drawn independently.
#' @param init character. Initialization distribution for embeddings:
#'  \describe{
#'    \item{`"uniform"`}{(default) Uniform distribution $U(-0.5/n_dims, 0.5/n_dims)$}
#'    \item{`"normal"`}{Standard normal distribution $N(0, 1)$}
#'  }
#' @param bootstrap_positive logical. If `TRUE`, positive samples are also drawn
#'  stochastically (with replacement). If `FALSE` (default), every observed
#'  co-occurrence is treated as exactly one positive example.
#' @param output character. What to return:
#'  \describe{
#'    \item{`"word_embeddings"`}{(default) only word embeddings (FCM rows)}
#'    \item{`"context_embeddings"`}{only context embeddings (FCM columns)}
#'    \item{`"all"`}{both word and context embeddings}
#'  }
#' @param seed integer. Random seed for reproducibility.
#' @param verbose logical. If `TRUE`, print progress information during training.
#' @param threads integer. Number of threads to use for training. Default is 
#'  `RcppParallel::defaultNumThreads()`.
#'
#' @details
#' The SGNS algorithm works by:
#' 1. Extracting co-occurrence counts from the FCM
#' 2. For each co-occurrence event, treating it as a positive (word, context) pair
#' 3. Sampling `neg` negative examples (random words) from the vocabulary
#' 4. Computing binary logistic loss for positive and negative pairs
#' 5. Updating embeddings via stochastic gradient descent with learning rate decay
#'
#' The algorithm learns two sets of embeddings:
#' - **Word embeddings**: Initialized randomly, updated primarily as predictors
#' - **Context embeddings**: Initialized randomly, updated primarily as targets
#'
#' For most applications, the word embeddings are the primary output. The
#' context embeddings can provide additional information or be used in an
#' ensemble. Set `output = "all"` to keep both.
#'
#' **Negative Sampling Strategy:**
#' Context frequencies are raised to the power of `context_smoothing` when constructing
#' the negative sampling table. The default `context_smoothing = 0.75` draws negative
#' samples proportional to $count(w)^{0.75}$, which has been empirically shown
#' to work better than uniform sampling (Mikolov et al., 2013). This balances
#' between rare and common words. Set `context_smoothing = 0` for uniform sampling.
#'
#' **Target Word Reweighting:**
#' The `target_smoothing` parameter controls how target words (FCM rows) are
#' weighted during training. Row sums are raised to the power of `target_smoothing`:
#' - `target_smoothing = 1` (default): Use observed frequencies (no reweighting)
#' - `target_smoothing = 0`: Uniform weighting (all words trained equally)
#' - `target_smoothing < 1`: Downsample frequent words
#' - `target_smoothing > 1`: Oversample frequent words
#'
#' **Subsampling:**
#' The `subsample` parameter implements the frequency-based subsampling from
#' Mikolov et al. (2013), which downweights very frequent pairs to improve
#' training efficiency and representation quality. Each pair (w, c) with
#' relative frequency f is kept with probability
#' $min(1, \sqrt{t/f} + t/f)$ where t is the `subsample` threshold.
#'
#' **Learning Rate Schedule:**
#' Learning rate decays linearly from `lr` to 0 across iterations:
#' $$\text{lr}(t) = \text{lr} \cdot (1 - \text{progress})$$
#' where progress ranges from 0 to 1. This schedule typically ensures convergence.
#'
#' @return A [dynamic_embeddings] object containing:
#'  \item{`word_embeddings`}{Matrix (or array) of word embedding vectors}
#'  \item{`context_embeddings`}{Matrix (or array) of context embedding vectors}
#'  \item{`fcm`}{The input FCM}
#'  \item{`control`}{Control parameters used}
#'  \item{`train_method`}{Character string "sgns"}
#'
#' @references
#' Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013).
#' Distributed representations of words and phrases and their compositionality.
#' In Advances in Neural Information Processing Systems (pp. 3111–3119).
#'
#' Goldberg, Y., & Levy, O. (2014). word2vec explained: Deriving Mikolov et al.'s
#' negative-sampling word-embedding method. arXiv preprint arXiv:1402.3722.
#'
#' @export
#' @examples
#' \dontrun{
#' # Create a simple FCM
#' toks <- quanteda::tokens(
#'   c("the quick brown fox", "the lazy dog", "quick fox"),
#'   remove_punct = TRUE
#' )
#' fcm_mat <- quanteda::fcm(toks, context = "window")
#'
#' # Train embeddings with SGNS
#' embeddings <- train_sgns(
#'   fcm_mat,
#'   n_dims = 50,
#'   neg = 5,
#'   iterations = 3,
#'   seed = 42L
#' )
#'
#' # Access the embeddings
#' head(embeddings$word_embeddings)
#' head(embeddings$context_embeddings)
#' }
train_sgns <- function(
  fcm,
  n_dims = 100,
  neg = 5,
  lr = 0.05,
  iterations = 5,
  batch_size = 1,
  context_smoothing = 0.75,
  target_smoothing = 1,
  subsample = 0,
  reject_positives = TRUE,
  init = "uniform",
  bootstrap_positive = FALSE,
  output = "word_embeddings",
  seed = NULL,
  verbose = TRUE,
  threads = RcppParallel::defaultNumThreads()
) {
  # Input validation
  stopifnot(
    "`fcm` must be a Quanteda fcm, sparseMatrix, SparseArray, or array" =
      (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") ||
       inherits(fcm, "SparseArray") || is.array(fcm)) &&
      (length(dim(fcm)) %in% c(2, 3)),
    "`n_dims` must be a positive integer" = is.numeric(n_dims) && n_dims > 0,
    "`neg` must be a positive integer" = is.numeric(neg) && neg > 0,
    "`lr` must be a positive number" = is.numeric(lr) && lr > 0,
    "`iterations` must be a positive integer" = is.numeric(iterations) && iterations > 0,
    "`batch_size` must be a positive integer" = is.numeric(batch_size) && batch_size > 0,
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
  iterations <- as.integer(iterations)
  batch_size <- as.integer(batch_size)
  threads <- as.integer(threads)

  # Handle 3D arrays
  if (length(dim(fcm)) == 3) {
    return(.train_sgns_3d(
      fcm, n_dims, neg, lr, iterations, batch_size,
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

  # Set number of threads for RcppParallel
  if (threads > 0) {
    RcppParallel::setThreadOptions(numThreads = threads)
  }

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
    n_iterations = iterations,
    batch_size = batch_size,
    smoothing = context_smoothing,
    reject_positives = reject_positives,
    init_type = init,
    bootstrap_positive = bootstrap_positive,
    seed = if (is.null(seed)) 0L else as.integer(seed),
    verbose = verbose
  )

  # Extract and name embeddings
  word_embeddings <- cpp_result$word_embeddings
  context_embeddings <- cpp_result$context_embeddings

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

  # Add SGNS-specific control information
  result$control <- list(
    method = "sgns",
    n_dims = n_dims,
    neg = neg,
    lr = lr,
    iterations = iterations,
    batch_size = batch_size,
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
.train_sgns_3d <- function(fcm, n_dims, neg, lr, iterations, batch_size,
                            context_smoothing, target_smoothing, subsample, reject_positives, init, 
                            bootstrap_positive, output, verbose, seed, threads) {
  fcm_ids <- dimnames(fcm)[[3]]
  fcm_list <- lapply(seq_len(dim(fcm)[3]), function(i) {
    train_sgns(
      fcm[, , i],
      n_dims = n_dims,
      neg = neg,
      lr = lr,
      iterations = iterations,
      batch_size = batch_size,
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

  # Preserve dimension names
  if (!is.null(dimnames(fcm)[[3]])) {
    if (!is.null(word_embeddings)) {
      dimnames(word_embeddings)[[3]] <- fcm_ids
    }
    if (!is.null(context_embeddings)) {
      dimnames(context_embeddings)[[3]] <- fcm_ids
    }
  }

  dynamic_embeddings(
    fcm = fcm,
    context_embeddings = context_embeddings,
    word_embeddings = word_embeddings,
    train_method = "sgns"
  )
}
