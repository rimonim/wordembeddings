#' Retrieve Word Embeddings
#'
#' Retrieve embeddings of particular words from a trained model. For the
#' [`"lm"`][train_lm()] and `"mlm"` training method,s embeddings may be computed
#' dynamically when requested.
#'
#' @param object a `dynamic_embeddings` object
#' @param newdata a character vector of words
#' @param drop logical. If `TRUE` (the default) and the result is one-dimensional
#'	(e.g. a single row), the output will be a (named) vector.
#' @param .keep_missing logical. What should be done about items in `newdata`
#'	that are not present in the embeddings object? If `FALSE` (the default),
#'	they will be ignored. If `TRUE`, they will be returned as `NA`.
#'
#'	@details
#'	Duplicated items in newdata will result in duplicated rows in the output.
#'	If an item in newdata matches multiple rows in object, the last one will be
#'	returned.

#' @rdname predict.dynamic_embeddings
#' @export
predict.dynamic_embeddings <- function(object, newdata, drop = TRUE, .keep_missing = FALSE){
	if (any(zchars <- !nzchar(newdata))) {
		warning(sprintf('Replacing %d empty strings with " ".', sum(zchars)))
		newdata[zchars] <- " "
	}
	embedding_not_found <- !(newdata %in% rownames(object$fcm))
	if (any(embedding_not_found)) {
		warning(sprintf("%d items in `newdata` are not present in the embeddings object.", sum(embedding_not_found)))
	}
	if (object$control$train_method == "lm") {
		out <- fit_word_embeddings_lm(fcm = object$fcm[newdata[!embedding_not_found],,drop = FALSE], context_embeddings = object$context_embeddings, control = object$control)
	}else if (object$control$train_method == "mlm") {
		out <- fit_word_embeddings_mlm(fcm = object$fcm[newdata[!embedding_not_found],,,drop = FALSE], context_embeddings = object$context_embeddings, control = object$control)
	}else{
		if (.keep_missing) {
			out <- matrix(nrow = length(newdata), ncol = ncol(object$word_embeddings), dimnames = list(newdata, colnames(object$word_embeddings)))
			out[!embedding_not_found,] <- object$word_embeddings[newdata[!embedding_not_found],]
			if (drop && nrow(out) == 1L) return(out[1,])
		}else{
			out <- object$word_embeddings[newdata[!embedding_not_found],,drop = drop]
			if (drop && length(out) == 0L) return(numeric())
		}
	}
	out
}
