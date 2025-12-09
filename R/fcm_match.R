#' Match the feature set of an feature co-occurence matrix to given feature names
#'
#' Match the feature set of a [Quanteda fcm][quanteda::fcm] (or similar matrix-like
#' object) to a specified vector of feature names. For existing features in x for
#' which there is an exact match for an element of features, these will be included.
#' Any features in x not features will be discarded, and any feature names specified
#' in features but not found in x will be added with all zero counts.
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or matrix-like object with row and column names
#' @param features a character vector of feature names
#' @param ... additional arguments (not used)
#'
#' @return A matrix-like object with row and column names matching the given feature names

#' @export
fcm_match <- function(fcm, features, ...) {
	UseMethod("fcm_match")
}

#' @export
fcm_match.default <- function(fcm, features, ...) {
	stopifnot(
		"fcm must have row and column names" = !(is.null(rownames(fcm)) | is.null(colnames(fcm))),
		"fcm must be a matrix or array object" = is.matrix(fcm) | inherits(fcm, c("SparseArray", "Matrix"))
	)

	fcm <- as(fcm, "SVT_SparseArray")

	missing_rownames <- setdiff(features, rownames(fcm))
	missing_colnames <- setdiff(features, colnames(fcm))
	fcm <- rbind(fcm, SparseArray::SVT_SparseArray(dim = c(length(missing_rownames), dim(fcm)[-1])))
	fcm <- cbind(fcm, SparseArray::SVT_SparseArray(dim = c(dim(fcm)[1], length(missing_colnames), dim(fcm)[-c(1:2)])))

	do.call(`[`, c(list(x = fcm, features, features), dimnames(fcm)[-c(1:2)]))
}

#' @export
fcm_match.fcm <- function(fcm, features, ...) {
	attrs <- attributes(fcm)
	fcm <- fcm_match.default(fcm, features, ...)
	fcm <- quanteda::as.fcm(as(fcm, "dgCMatrix"))
	fcm@meta <- attrs[["meta"]]
	fcm
}

#' @export
fcm_match.matrix <- function(fcm, features, ...) {
	fcm <- fcm_match.default(fcm, features, ...)
	as.matrix(fcm)
}
