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
#' @param output character. The default, `"word_embeddings"` includes only word
#'	embeddings in the output. `"context_embeddings"` includes only context
#'	embeddings. `"all"` includes both word and context embeddings.

#' @export
train_svd <- function(fcm, n_dims = 100, eig = 0, output = "word_embeddings") {
	include_word_embeddings <- output %in% c("word_embeddings", "all")
	include_context_embeddings <- output %in% c("context_embeddings", "all")
	nv <- 0
	nu <- 0
	if (include_word_embeddings) nu <- n_dims
	if (include_context_embeddings) nv <- n_dims
	svd <- RSpectra::svds(fcm, k = n_dims, nu = nu, nv = nv)
	if (length(eig) == 2) {
		if (include_word_embeddings) word_embeddings <- svd$u %*% diag(svd$d^eig[1]) else word_embeddings <- NULL
		if (include_context_embeddings) context_embeddings <- t( diag(svd$d^eig[2]) %*% t(svd$v) ) else context_embeddings <- NULL
	}else if (length(eig) == 1) {
      	if (include_word_embeddings) word_embeddings <- svd$u %*% diag(svd$d^eig) else word_embeddings <- NULL
		if (include_context_embeddings) context_embeddings <- t( diag(svd$d^eig) %*% t(svd$v) ) else context_embeddings <- NULL
	}else{
		stop("eig must be of length 1 or 2")
	}
  	if (include_word_embeddings) rownames(word_embeddings) <- rownames(fcm)
	if (include_context_embeddings) rownames(context_embeddings) <- colnames(fcm)
	dynamic_embeddings(fcm = fcm, context_embeddings = context_embeddings, word_embeddings = word_embeddings, train_method = "svd")
}

#' @rdname train_svd
#' @export
train_svd_context <- function(fcm, n_dims = 100, eig = 0) {
	train_svd(fcm = fcm, n_dims = n_dims, eig = 0, output = "context_embeddings")$context_embeddings
}
