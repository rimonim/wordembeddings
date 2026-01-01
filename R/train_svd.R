#' Train embeddings with SVD
#'
#' `train_svd(fcm)` decomposes the input `fcm` to produce word and/or context
#' embeddings. `train_svd_context(fcm)` is equivalent to
#' `train_svd(fcm, output = "context_embeddings")$context_embeddings`.
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D
#'	array-like object
#' @param n_dims integer. Dimensionality of embeddings.
#' @param eig numeric. Exponent for weighting the eigenvalue matrix, as per
#'	Caron (2001). The default, `0`, corresponds to ignoring the eigenvalues
#'	entirely.
#' @param row_weights numeric vector. Optional weights for the rows of `fcm`.
#' @param col_weights numeric vector. Optional weights for the columns of `fcm`.
#' @param output character. The default, `"word_embeddings"` includes only word
#'	embeddings in the output. `"context_embeddings"` includes only context
#'	embeddings. `"all"` includes both word and context embeddings.
#' @param opts control parameters passed to [RSpectra::svds()].

#' @export
train_svd <- function(
	fcm, 
	n_dims = 100, 
	eig = 0, 
	row_weights = NULL, 
	col_weights = NULL, 
	output = "word_embeddings",
	opts = list()
) {
	include_word_embeddings <- output %in% c("word_embeddings", "all")
	include_context_embeddings <- output %in% c("context_embeddings", "all")
	nv <- 0
	nu <- 0
	if (include_word_embeddings) nu <- n_dims
	if (include_context_embeddings) nv <- n_dims
	# weighted SVD if weights provided
	if (!is.null(row_weights)) {
		stopifnot(
			"length(row_weights) must equal nrow(fcm)" = length(row_weights) == nrow(fcm),
			"row_weights must be non-negative" = all(row_weights >= 0)
		)
		fcm <- fcm * sqrt(row_weights)
	}

	if (!is.null(col_weights)) {
		stopifnot(
			"length(col_weights) must equal ncol(fcm)" = length(col_weights) == ncol(fcm),
			"col_weights must be non-negative" = all(col_weights >= 0)
		)
		fcm <- t(t(fcm) * sqrt(col_weights))
	}
	# compute SVD
	svd <- RSpectra::svds(fcm, k = n_dims, nu = nu, nv = nv, opts = opts)
	if (length(eig) == 2) {
		r_eig <- eig[1]
		c_eig <- eig[2]
	}else if (length(eig) == 1) {
		r_eig <- eig
		c_eig <- eig
	}else{
		stop("eig must be of length 1 or 2")
	}
	if (include_word_embeddings) {
		if (r_eig != 0) {
			word_embeddings <- svd$u %*% diag(svd$d^r_eig)
		} else {
			word_embeddings <- svd$u
		}
	}else{
		word_embeddings <- NULL
	}
	if (include_context_embeddings) {
		if (c_eig != 0) {
			context_embeddings <- t( diag(svd$d^c_eig) %*% t(svd$v) ) 
		} else {
			context_embeddings <- svd$v
		}
	}else{
		context_embeddings <- NULL
	}
  if (include_word_embeddings) rownames(word_embeddings) <- rownames(fcm)
	if (include_context_embeddings) rownames(context_embeddings) <- colnames(fcm)
	list(word_embeddings = word_embeddings, context_embeddings = context_embeddings)
}

#' @rdname train_svd
#' @export
train_svd_context <- function(fcm, n_dims = 100, eig = 0) {
	train_svd(fcm = fcm, n_dims = n_dims, eig = 0, output = "context_embeddings")$context_embeddings
}
