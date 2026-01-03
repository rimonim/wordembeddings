#' Train embeddings with GloVe
#'
#' `train_glove()` trains word and context embeddings using the GloVe 
#' (Global Vectors) algorithm. GloVe is a weighted matrix factorization method
#' that learns embeddings by factorizing the log co-occurrence matrix.
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D
#'   array-like object containing co-occurrence counts.
#' @param n_dims integer. Dimensionality of embeddings.
#' @param x_max numeric. Maximum co-occurrence count for weighting function.
#'   Co-occurrences above this value receive weight 1.0.
#' @param alpha numeric. Exponent for the weighting function. Default is 0.75
#'   as in the original GloVe paper.
#' @param lr numeric. Learning rate for AdaGrad optimizer. Default is 0.05.
#' @param epochs integer. Number of training epochs. Default is 15.
#' @param weight_fn character. Weighting function type:
#'   \describe{
#'     \item{"glove"}{(default) Original GloVe weighting: `f(x) = (x/x_max)^alpha` 
#'       if `x < x_max`, else 1.}
#'     \item{"power"}{Power weighting: `f(x) = x^alpha` (no cap).}
#'     \item{"log"}{Log weighting: `f(x) = min(1, log(x)/log(x_max))`.}
#'     \item{"uniform"}{Uniform weighting: `f(x) = 1`.}
#'   }
#' @param fix_bias logical. If `TRUE`, fix the word and context biases at
#'   `log(marginal_count)` rather than learning them. This can be useful when
#'   you want the bias terms to exactly capture marginal frequencies.
#' @param output character. Which embeddings to include in the output:
#'   \describe{
#'     \item{"word_embeddings"}{(default) Include only word embeddings.}
#'     \item{"context_embeddings"}{Include only context embeddings.}
#'     \item{"all"}{Include both word and context embeddings.}
#'   }
#' @param init character. Initialization method for embeddings:
#'   \describe{
#'     \item{"uniform"}{(default) Uniform random in `[-0.5/n_dims, 0.5/n_dims]`.}
#'     \item{"normal"}{Normal random with mean 0 and sd 0.01.}
#'   }
#' @param shuffle logical. Whether to shuffle co-occurrence pairs each epoch.
#'   Default is `TRUE`.
#' @param seed integer. Random seed for reproducibility.
#' @param verbose logical. Print progress information.
#' @param threads integer. Number of threads. Default uses all available cores.
#'
#' @return A list containing:
#'   \describe{
#'     \item{word_embeddings}{Matrix of word embeddings (nrow(fcm) x n_dims),
#'       if requested.}
#'     \item{context_embeddings}{Matrix of context embeddings (ncol(fcm) x n_dims),
#'       if requested.}
#'     \item{bias_i}{Word bias terms.}
#'     \item{bias_j}{Context bias terms.}
#'     \item{cost_history}{Loss value at each epoch.}
#'   }
#'
#' @references
#' Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors
#' for Word Representation. In Proceedings of the 2014 Conference on Empirical
#' Methods in Natural Language Processing (EMNLP) (pp. 1532-1543).
#'
#' @examples
#' \dontrun{
#' # Create a simple co-occurrence matrix
#' library(quanteda)
#' toks <- tokens(c("the cat sat on the mat", "the dog ran in the park"))
#' fcm_mat <- fcm(toks, context = "window", window = 2)
#'
#' # Train GloVe embeddings
#' result <- train_glove(fcm_mat, n_dims = 50, epochs = 10)
#'
#' # Access embeddings
#' word_emb <- result$word_embeddings
#' }
#'
#' @export
train_glove <- function(
	fcm,
	n_dims = 100,
	x_max = 100,
	alpha = 0.75,
	lr = 0.05,
	epochs = 15,
	weight_fn = c("glove", "power", "log", "uniform"),
	fix_bias = FALSE,
	output = c("word_embeddings", "context_embeddings", "all"),
	init = c("uniform", "normal"),
	shuffle = TRUE,
	seed = NULL,
	verbose = TRUE,
	threads = parallel::detectCores()
) {
	# Match arguments
	weight_fn <- match.arg(weight_fn)
	output <- match.arg(output)
	init <- match.arg(init)
	
	# Input validation
	stopifnot(
		"`fcm` must be a Quanteda fcm, sparseMatrix, SparseArray, or array" =
			(inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") ||
			 inherits(fcm, "SparseArray") || is.array(fcm)) &&
			(length(dim(fcm)) %in% c(2, 3)),
		"`n_dims` must be a positive integer" = is.numeric(n_dims) && n_dims > 0,
		"`x_max` must be a positive number" = is.numeric(x_max) && x_max > 0,
		"`alpha` must be a non-negative number" = is.numeric(alpha) && alpha >= 0,
		"`lr` must be a positive number" = is.numeric(lr) && lr > 0,
		"`epochs` must be a positive integer" = is.numeric(epochs) && epochs > 0,
		"`fix_bias` must be logical" = is.logical(fix_bias),
		"`shuffle` must be logical" = is.logical(shuffle),
		"`verbose` must be logical" = is.logical(verbose),
		"`threads` must be a positive integer" = is.numeric(threads) && threads > 0
	)
	
	if (is.null(seed)) seed <- 1L
	
	n_dims <- as.integer(n_dims)
	epochs <- as.integer(epochs)
	threads <- as.integer(threads)
	
	# Determine which embeddings to output
	include_word_embeddings <- output %in% c("word_embeddings", "all")
	include_context_embeddings <- output %in% c("context_embeddings", "all")
	
	# Handle 3D arrays
	if (length(dim(fcm)) == 3) {
		return(.train_glove_3d(
			fcm, n_dims, x_max, alpha, lr, epochs, weight_fn, fix_bias,
			include_word_embeddings, include_context_embeddings,
			init, shuffle, seed, verbose, threads
		))
	}
	
	# Convert to sparse triplet format
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
	row_vocab <- rownames(fcm_sparse)
	col_vocab <- colnames(fcm_sparse)
	if (is.null(row_vocab)) {
		row_vocab <- paste0("word_", seq_len(nrow(fcm_sparse)))
	}
	if (is.null(col_vocab)) {
		col_vocab <- paste0("context_", seq_len(ncol(fcm_sparse)))
	}
	
	# Compute marginal sums for fix_bias
	if (fix_bias) {
		row_sums <- Matrix::rowSums(fcm_sparse)
		col_sums <- Matrix::colSums(fcm_sparse)
	} else {
		row_sums <- numeric(nrow(fcm_sparse))
		col_sums <- numeric(ncol(fcm_sparse))
	}
	
	# Call C++ implementation
	result <- glove_fit_cpp(
		i_indices = fcm_sparse@i,
		j_indices = fcm_sparse@j,
		x_values = fcm_sparse@x,
		n_rows = nrow(fcm_sparse),
		n_cols = ncol(fcm_sparse),
		n_dims = n_dims,
		x_max = x_max,
		alpha = alpha,
		lr = lr,
		epochs = epochs,
		weight_type_str = weight_fn,
		fix_bias = fix_bias,
		row_sums = row_sums,
		col_sums = col_sums,
		init_type = init,
		seed = as.integer(seed),
		verbose = verbose,
		shuffle = shuffle,
		threads = threads,
		include_word_embeddings = include_word_embeddings,
		include_context_embeddings = include_context_embeddings
	)
	
	# Add row names to embeddings
	if (include_word_embeddings && !is.null(result$word_embeddings)) {
		rownames(result$word_embeddings) <- row_vocab
	}
	if (include_context_embeddings && !is.null(result$context_embeddings)) {
		rownames(result$context_embeddings) <- col_vocab
	}
	
	# Add names to biases
	names(result$bias_i) <- row_vocab
	names(result$bias_j) <- col_vocab
	
	result
}

#' @keywords internal
#' Handle 3D FCM arrays
.train_glove_3d <- function(
	fcm, n_dims, x_max, alpha, lr, epochs, weight_fn, fix_bias,
	include_word_embeddings, include_context_embeddings,
	init, shuffle, seed, verbose, threads
) {
	fcm_ids <- dimnames(fcm)[[3]]
	
	output <- if (include_word_embeddings && include_context_embeddings) {
		"all"
	} else if (include_context_embeddings) {
		"context_embeddings"
	} else {
		"word_embeddings"
	}
	
	fcm_list <- lapply(seq_len(dim(fcm)[3]), function(i) {
		train_glove(
			fcm[, , i],
			n_dims = n_dims,
			x_max = x_max,
			alpha = alpha,
			lr = lr,
			epochs = epochs,
			weight_fn = weight_fn,
			fix_bias = fix_bias,
			output = output,
			init = init,
			shuffle = shuffle,
			seed = seed,
			verbose = verbose && i == 1,  # Only verbose for first slice
			threads = threads
		)
	})
	
	# Combine results
	result <- list()
	
	if (include_word_embeddings) {
		word_emb_list <- lapply(fcm_list, `[[`, "word_embeddings")
		result$word_embeddings <- S4Arrays::abind(word_emb_list, along = 3)
		if (!is.null(fcm_ids)) {
			dimnames(result$word_embeddings)[[3]] <- fcm_ids
		}
	}
	
	if (include_context_embeddings) {
		context_emb_list <- lapply(fcm_list, `[[`, "context_embeddings")
		result$context_embeddings <- S4Arrays::abind(context_emb_list, along = 3)
		if (!is.null(fcm_ids)) {
			dimnames(result$context_embeddings)[[3]] <- fcm_ids
		}
	}
	
	# Combine biases into matrices
	result$bias_i <- do.call(cbind, lapply(fcm_list, `[[`, "bias_i"))
	result$bias_j <- do.call(cbind, lapply(fcm_list, `[[`, "bias_j"))
	if (!is.null(fcm_ids)) {
		colnames(result$bias_i) <- fcm_ids
		colnames(result$bias_j) <- fcm_ids
	}
	
	# Average cost history
	result$cost_history <- Reduce(`+`, lapply(fcm_list, `[[`, "cost_history")) / length(fcm_list)
	
	result
}
