#' Train embeddings with multilevel linear regression
#'
#' This training paradigm follows the same principle as `train_lm()`, but
#' operates on a stacked (3D) feature co-occurrence matrix representing
#' co-occurrence patterns in a set of communities or individuals. Each
#' community or individual has distinct word embeddings, but these are assumed
#' to come from a single underlying normal distribution and therefore
#' information can be efficiently shared between communities or individuals
#' using partial pooling.
#'
#' @param fcm a 3D array-like object
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

#' @export
train_mlm <- function(fcm, fcm_weight = NULL, n_dims = 100, context_embeddings = train_svd_context, ..., dynamic = TRUE, control = dynamic_embeddings_control()) {
	if (is.function(context_embeddings)) {
		cat("Training context embeddings...", sep = "\n")
		context_embeddings <- context_embeddings(fcm, n_dims = n_dims, ...)
	}
	if (!dynamic) {
		cat("Training word embeddings...", sep = "\n")
		word_embeddings <- fit_word_embeddings_mlm(fcm = fcm, fcm_weight = fcm_weight, context_embeddings = context_embeddings, control = control)
	}else{
		word_embeddings <- NULL
	}
	dynamic_embeddings(fcm = fcm, context_embeddings = context_embeddings, word_embeddings = word_embeddings, dynamic_embeddings_control(control, train_method = "mlm"))
}

#' @rdname fit_word_embeddings
#' @keywords internal
fit_word_embeddings_mlm <- function(fcm, fcm_weight = NULL, context_embeddings, control = dynamic_embeddings_control()) {
	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (inherits(fcm, "sparseMatrix") || inherits(fcm, "fcm") || inherits(fcm, "SparseArray") || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3)),
		"fcm cols and context_embeddings rows must match" = ncol(fcm) == nrow(context_embeddings) && (is.null(rownames(context_embeddings)) || is.null(colnames(fcm)) || all(colnames(context_embeddings) == rownames(fcm))),
		"fcm_weight must have the same dimensions as fcm" = is.null(fcm_weight) || all(dim(fcm) == dim(fcm_weight)),
		"fcm must be 3D for mlm fitting. Please use lm instead." = length(dim(fcm)) == 3
	)
	control <- dynamic_embeddings_control(control, train_method = "mlm")
	if (nrow(fcm) >= 500) {
		pb <- utils::txtProgressBar(0, nrow(fcm), style = 3)
		on.exit(close(pb), add = TRUE)
	}
	context_embeddings_df <- as.data.frame(context_embeddings)
	context_embeddings_df <- as.data.frame(lapply(context_embeddings_df, function(v) rep(v, times = dim(fcm)[3])))
	context_embeddings_df$ID <- factor(rep(dimnames(fcm)[[3]], each = ncol(fcm)))

	# set model formula
	model_formula <- setdiff(names(context_embeddings_df), c("ID"))
	if (control$mlm_engine == "mgcv") {
		model_formula <- c("0", model_formula, paste0("s(ID,",model_formula, ",bs='re')"))
	}else{
		model_formula <- c(model_formula, paste0("(",paste(model_formula, collapse = "+"), "|ID)"))
	}
	model_formula <- reformulate(model_formula, response = "y")
	if (control$mlm_engine == "jglmm") {
		if (!requireNamespace("jglmm", quietly = TRUE)) stop("mlm fitting requires jglmm package. Install using install.packages('jglmm')")
		jglmm::jglmm_setup()
	}

	# fit models
	clust <- parallel::makeForkCluster()
	if (is.null(fcm_weight)) {
		word_embeddings <- lapply(seq_len(nrow(fcm)), function(r) {
			if (nrow(fcm) >= 500 && r %% 50 == 0) utils::setTxtProgressBar(pb, r)
			context_embeddings_df$y <- as.vector(fcm[r,,])
			if (control$mlm_engine == "jglmm") {
				if (control$drop_zeros) {
					mod <- try(jglmm::jglmm(model_formula, context_embeddings_df[context_embeddings_df$y != 0,]))
				}else{
					mod <- try(jglmm::jglmm(model_formula, context_embeddings_df))
				}
				if (inherits(mod, "try-error")) return(matrix(nrow = dim(fcm)[3], ncol = ncol(context_embeddings)))
				JuliaCall::julia_assign("model", mod$model)
				out <- JuliaCall::julia_eval("model_ranef = map(DataFrame, raneftables(model));")[[1]]
				out_rownames <- out[,"ID"]
				out <- as.matrix(out[,-1])
				rownames(out) <- out_rownames
				return(out)
			}else if (control$mlm_engine == "mgcv") {
				if (!requireNamespace("mgcv", quietly = TRUE)) stop("mlm fitting requires mgcv package. Install using install.packages('mgcv')")
				if (control$drop_zeros) {
					coefs <- try(stats::coef(mgcv::bam(model_formula, data = context_embeddings_df[context_embeddings_df$y != 0,], cluster = clust)))
				}else{
					coefs <- try(stats::coef(mgcv::bam(model_formula, data = context_embeddings_df, cluster = clust)))
				}
				if (inherits(coefs, "try-error")) return(matrix(nrow = dim(fcm)[3], ncol = ncol(context_embeddings)))
				ID <- stringr::str_extract(names(coefs), "(?<=\\)\\.)[:digit:]+")
				coefs <- coefs[!is.na(ID)] + rep(coefs[is.na(ID)], each = dim(fcm)[3]) # add fixed effects to random effects
				return(matrix(coefs, dim(fcm)[3]))
			}else if (control$mlm_engine == "lme4") {
				if (!requireNamespace("lme4", quietly = TRUE)) stop("mlm fitting requires lme4 package. Install using install.packages('lme4')")
				if (control$drop_zeros) {
					coefs <- try(stats::coef(lme4::lmer(model_formula, data = context_embeddings_df[context_embeddings_df$y != 0,], control = lme4::lmerControl(calc.derivs = FALSE)))[[1]])
				}else{
					coefs <- try(stats::coef(lme4::lmer(model_formula, data = context_embeddings_df, control = lme4::lmerControl(calc.derivs = FALSE)))[[1]])
				}
				if (inherits(coefs, "try-error")) return(matrix(nrow = dim(fcm)[3], ncol = ncol(context_embeddings)))
			}else{
				stop("Invalid engine. Please specify either 'mgcv', 'lme4', or 'jglmm'.")
			}
		})
	}else{
		word_embeddings <- lapply(seq_len(nrow(fcm)), function(r) {
			if (nrow(fcm) >= 500 && r %% 50 == 0) utils::setTxtProgressBar(pb, r)
			context_embeddings_df$y <- as.vector(fcm[r,,])
			w <- as.vector(fcm_weight[r,,])
			if (control$mlm_engine == "jglmm") {
				if (control$drop_zeros) {
					mod <- jglmm::jglmm(model_formula, context_embeddings_df[context_embeddings_df$y != 0,], weights = w[context_embeddings_df$y != 0])
				}else{
					mod <- jglmm::jglmm(model_formula, context_embeddings_df, weights = w)
				}
				JuliaCall::julia_assign("model", mod$model)
				out <- JuliaCall::julia_eval("model_ranef = map(DataFrame, raneftables(model));")[[1]]
				out_rownames <- out[,"ID"]
				out <- as.matrix(out[,-c(1,2)])
				rownames(out) <- out_rownames
				return(out)
			}else if (control$mlm_engine == "lme4") {
				if (control$drop_zeros) {
					return(matrix(stats::coef(lme4::lmer(model_formula, weights = w[context_embeddings_df$y != 0], data = context_embeddings_df[context_embeddings_df$y != 0,], control = lme4::lmerControl(calc.derivs = FALSE)))[[1]], dim(fcm)[3]))
				}else{
					return(matrix(stats::coef(lme4::lmer(model_formula, weights = w, data = context_embeddings_df, control = lme4::lmerControl(calc.derivs = FALSE)))[[1]], dim(fcm)[3]))
				}
			}else{
				stop("Invalid engine. Please specify either 'mgcv', lme4', or 'jglmm'.")
			}
		})
	}
	out <- array(dim = c(ncol(fcm), ncol(context_embeddings), dim(fcm)[3]))
	for (id in seq_along(word_embeddings)) {
		out[id,,] <- word_embeddings[[id]]
	}
	# out <- do.call(S4Arrays::abind, c(word_embeddings, along = 3))
	rownames(out) <- rownames(fcm)
	out
}
