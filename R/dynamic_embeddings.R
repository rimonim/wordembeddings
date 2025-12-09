#' Dynamic Embeddings Class
#'
#' A dynamic_embeddings object is a list containing embeddings and/or
#' a specification of the embedding method. This allows methods such as
#' `predict()`, which may in some cases compute embeddings dynamically to
#' avoid maintaining too bulky a model object. dynamic_embeddings objects are
#' generally produced as the output of a `train_` function (e.g. `train_svd()`)
#' rather than being constructed directly.
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D
#'	array-like object
#' @param context_embeddings a matrix or array with the same number of rows as `fcm`
#' @param word_embeddings a matrix or array with the same number of rows as `fcm`
#' @param control a list of control parameters created by `dynamic_embeddings_control()`
#' @param train_method the method used to train the embeddings
#'
#' @export
dynamic_embeddings <- function(fcm = array(), context_embeddings = array(), word_embeddings = NULL, control = dynamic_embeddings_control(), train_method = NULL) {
	if (!is.null(word_embeddings)) {
    word_embeddings <- as.array(word_embeddings)
  }
  if (!is.null(context_embeddings)) {
		if (length(dim(context_embeddings)) == 2) {
			context_embeddings <- as.matrix(context_embeddings)
		}else{
			context_embeddings <- as.array(context_embeddings)
		}
  }

	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") || inherits(fcm, "SparseArray") || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3)),
		"context_embeddings and fcm must have the same number of rows" = nrow(fcm) == nrow(context_embeddings),
		"context_embeddings rownames must match fcm rownames" = is.null(rownames(fcm)) || is.null(rownames(context_embeddings)) || all(rownames(fcm) == rownames(context_embeddings))
	)

	structure(list(fcm = fcm, context_embeddings = context_embeddings, word_embeddings = word_embeddings, control = control, train_method = train_method), class = "dynamic_embeddings")
}
