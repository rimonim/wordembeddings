#' Reduce a 3-D array to a 2-D feature co-occurrence matrix
#'
#' Reduce a N × N × M (sparse) array to an N × N matrix by summing (or some
#' other function) along the third dimension. Thus if `fcm` represents a stack
#' of feature co-occurrence matrices (FCMs), `fcm_reduce()` will compute the
#' aggregate FCM.
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D array-like object
#' @param f a binary aggregation function to be passed to `base::Reduce()`
#'
#' @export
fcm_reduce <- function(fcm, f = `+`) {
	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") || inherits(fcm, "SparseArray") || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3))
	)
	if (length(dim(fcm)) == 2) return(fcm)
	Reduce(f, lapply(seq_len(dim(fcm)[3]), function(k) fcm[,,k]))
}
