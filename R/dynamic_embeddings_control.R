#' Control Parameters for Dynamic Embeddings Objects
#'
#' @param control (optional) an existing list of control parameters to modify
#' @param train_method the method used to train the embeddings; one of `"svd"`,
#'	`"mds"`, `"sgns"`, `"glove"`, `"lm"`, or `"mlm"`.
#' @param drop_zeros when `train_method = "lm"` or `"mlm"` should zeros be
#'	dropped from fcm while fitting the model?
#' @param mlm_engine engine for fitting random effects models when
#'	`train_method = "mlm"`. Either `"mgcv"` (the default), `"lme4"`, or `"jglmm"`
#'	(requires Julia and the `jglmm` package).

#' @importFrom rlang `%||%`
#' @export
dynamic_embeddings_control <- function(control = NULL, train_method = NULL, drop_zeros = NULL, mlm_engine = NULL) {
	if (!is.null(control) && any(!(c("train_method", "drop_zeros", "mlm_engine") %in% names(control)))) {
		warning("Existing control specification is missing expected parameters.")
	}
	if (is.null(control)) control <- list(train_method = train_method, drop_zeros = drop_zeros, mlm_engine = mlm_engine)
	control <- list(
		train_method = train_method %||% control$train_method %||% train_method,
		drop_zeros = drop_zeros %||% control$drop_zeros %||% FALSE,
		mlm_engine = mlm_engine %||% control$mlm_engine %||% if (is.null(train_method)) NULL else if (train_method == "mlm") "mgcv" else NULL
	)
	control
}
