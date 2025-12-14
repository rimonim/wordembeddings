#' Train embeddings with linear regression
#'
#' Given a set of context embeddings and a feature co-occurence matrix (FCM),
#' an embedding for a given word can be trained using multiple linear regression
#' with the context embedding dimensions as predictors and the corresponding row
#' of the FCM as the dependent variable. This method enables the quantification
#' of uncertainty around the resulting embeddings (Vallebueno et al., 2024).
#'
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D
#'	array-like object
#' @param fcm_weight (optional) an array-like object of the same dimensionality
#'	as `fcm`, giving weights for the regression.
#' @param n_dims integer. Dimensionality of embeddings. Ignored if
#'	`context_embeddings` is a matrix or array.
#' @param context_embeddings either a matrix or array with the same number of
#'	rows as `fcm`, or a function that takes an FCM and outputs such a matrix or
#'	array.
#' @param ... additional parameters to be passed to `context_embeddings` function
#' @param dynamic logical. `TRUE` (the default) produces a `dynamic_embeddings`
#'	object from which word embeddings can be computed dynamically as needed.
#' `FALSE` computes all word embeddings upfront.
#' @param control a list of control parameters created by `dynamic_embeddings_control()`
#'
#' @details
#' For GloVe-V (Vallebueno et al., 2024), `fcm` should be a matrix of log
#' co-occurrence counts with pretrained word and context biases subtracted out,
#' `fcm_weight` should be a matrix of transformed co-occurrence counts such that
#' low frequencies are devalued, and `context_embeddings` should be pretrained
#' glove context embeddings.
#'
#' @references Vallebueno, A., Handan-Nader, C., Manning, C. D., & Ho, D. E. (2024). Statistical Uncertainty in Word Embeddings: GloVe-V (arXiv:2406.12165). arXiv. https://doi.org/10.48550/arXiv.2406.12165

#' @export
train_lm <- function(fcm, fcm_weight = NULL, n_dims = 100, context_embeddings = train_svd_context, ..., dynamic = TRUE, control = dynamic_embeddings_control()) {
	if (is.function(context_embeddings)) {
		cat("Training context embeddings...", sep = "\n")
		context_embeddings <- context_embeddings(fcm, n_dims = n_dims, ...)
	}
	if (!dynamic) {
		cat("Training word embeddings...", sep = "\n")
		word_embeddings <- fit_word_embeddings_lm(fcm = fcm, fcm_weight = fcm_weight, context_embeddings = context_embeddings, control = control)
	}else{
		word_embeddings <- NULL
	}
	dynamic_embeddings(fcm = fcm, context_embeddings = context_embeddings, word_embeddings = word_embeddings, control = dynamic_embeddings_control(control, train_method = "lm"))
}


#' @rdname fit_word_embeddings
#' Helpers to fit word embeddings using regression
#' @keywords internal
fit_word_embeddings_lm <- function(fcm, fcm_weight = NULL, context_embeddings, control = dynamic_embeddings_control()) {
	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") || inherits(fcm, "SparseArray") || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3)),
		"fcm cols and context_embeddings rows must match" = ncol(fcm) == nrow(context_embeddings) && (is.null(rownames(context_embeddings)) || is.null(colnames(fcm)) || all(rownames(context_embeddings) == colnames(fcm))),
		"fcm_weight must have the same dimensions as fcm" = is.null(fcm_weight) || all(dim(fcm) == dim(fcm_weight))
	)
	control <- dynamic_embeddings_control(control, train_method = "lm")
	if (nrow(fcm) >= 500) {
		pb <- utils::txtProgressBar(0, nrow(fcm), style = 3)
		on.exit(close(pb), add = TRUE)
	}
	context_embeddings_df <- as.data.frame(context_embeddings)
	if (is.null(fcm_weight)) {
		word_embeddings <- lapply(seq_len(nrow(fcm)), function(r) {
			if (nrow(fcm) >= 500 && r %% 50 == 0) utils::setTxtProgressBar(pb, r)
			y <- fcm[r,]
			if (control$drop_zeros) {
				return(stats::lm(y[y != 0] ~ 0 + ., data = context_embeddings_df[y != 0,])$coefficients)
			}else{
				return(stats::lm(y ~ 0 + ., data = context_embeddings_df)$coefficients)
			}
		})
	}else{
		word_embeddings <- lapply(seq_len(nrow(fcm)), function(r) {
			if (nrow(fcm) >= 500 && r %% 50 == 0) utils::setTxtProgressBar(pb, r)
			y <- fcm[r,]
			if (control$drop_zeros) {
				return(stats::lm(y[y != 0] ~ 0 + ., weights = fcm_weight[r,][y != 0], data = context_embeddings_df[y != 0,])$coefficients)
			}else{
				return(stats::lm(y ~ 0 + ., weights = fcm_weight[r,], data = context_embeddings_df)$coefficients)
			}
		})
	}
	word_embeddings <- do.call(rbind, word_embeddings)
	rownames(word_embeddings) <- rownames(fcm)
	word_embeddings
}
