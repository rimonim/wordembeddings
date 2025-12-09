#' Transform a feature co-occurrence matrix
#'
#' `fcm_pmi()` calculates the Pointwise Mutual Information between each word and
#' context (i.e. row and column) of the matrix. `fcm_log()` takes the logarithm
#' of each element. `fcm_smooth()` re-weights observed co-occurrence frequencies
#' to account for unobserved values. `fcm_positive()` sets all negative values to
#' zero.
#'
#' @rdname fcm_transformations
#' @param fcm a [Quanteda fcm][quanteda::fcm] or similar 2D matrix-like or 3D array-like object
#' @param base the base with which to compute logarithms
#' @param positive logical. If `TRUE`, all negative elements are replaced with zeros.
#' @param context_smoothing numeric. Power to raise context (column) frequencies. Default is 1.
#'  Mikolov et al. (2013) found 0.75 to work well for word2vec.
#' @param target_smoothing numeric. Power to raise target word (row) frequencies. Default is 1.
#'  Set to 0 for uniform target weighting.
#' @param shift numeric. A number added to each element of the output (see details).
#' @param prob `"rows"` indicates that elements are row probabilities (i.e. row sums
#' are assumed to be 1). `"cols"` indicates that elements are column probabilities
#' (i.e. column sums are assumed to be 1). If `NULL` (the default), elements are
#' taken as raw counts.
#'
#' @details
#' If `fcm` is a 3D array or SparseArray, it is taken as a stack of feature
#' co-occurrence matrices. In other words, PMI is calculated for rows and
#' columns within each level of the third dimension.
#'
#' **Smoothing Parameters:**
#' Both `context_smoothing` and `target_smoothing` control how frequencies are
#' weighted when computing PMI. Context counts (columns) are raised to the power
#' of `context_smoothing`, and target word counts (rows) are raised to the power
#' of `target_smoothing`. Mikolov et al. (2013) found `context_smoothing = 0.75`
#' to work well for word2vec. Setting either parameter to 0 results in uniform
#' weighting for that dimension.
#'
#' The shift parameter controls the prior probability of observing a true
#' co-occurence as opposed to a randomly sampled ("negative") one.
#' Thus `shift = -log(k, base)` is analogous to the Skip-gram algorithm with k
#' negative samples. `shift = log(sum(fcm), base)` is analogous to the GloVe
#' algorithm with weights fixed at log frequency.
#'
#' @references
#' Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.
#'

#' @export
fcm_pmi <- function(fcm, positive = TRUE, context_smoothing = 1, target_smoothing = 1, 
                    shift = 0, base = 2, prob = NULL) {
	if( is_quanteda <- inherits(fcm, "fcm") ) {
		fcm_meta <- fcm@meta
	}
	is_sparseMatrix <- inherits(fcm, "sparseMatrix")
	is_SparseArray <- inherits(fcm, "SparseArray")
	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (is_sparseMatrix || is_quanteda || is_SparseArray || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3))
	)
	if (any(fcm < 0)) warning("fcm includes negative values. Treating these as zero in calculation...")
	if (length(dim(fcm)) == 3) {
		fcm_ids <- dimnames(fcm)[[3]]
		fcm <- lapply(seq_len(dim(fcm)[3]), function(i) {
			fcm_pmi(fcm[,,i], positive = positive, context_smoothing = context_smoothing, 
			        target_smoothing = target_smoothing, shift = shift, base = base)
		})
		fcm <- do.call(S4Arrays::abind, c(fcm, along = 3))
		dimnames(fcm)[[3]] <- fcm_ids
		return(fcm)
	}

	# convert to TsparseMatrix if necessary
	if (is_quanteda || is_sparseMatrix || is_SparseArray) {
		fcm <- methods::as(fcm, "TsparseMatrix")
	}else{
		fcm <- as.matrix(fcm)
	}

	# precompute
	total_count <- sum(fcm)
	if (is.null(prob)) {
		row_sums <- Matrix::rowSums(fcm)
		col_sums <- Matrix::colSums(fcm)
	}else if (prob == "rows") {
		row_sums <- rep(1, nrow(fcm))
		col_sums <- Matrix::colSums(fcm) * nrow(fcm)
	}else if (prob == "cols") {
		row_sums <- Matrix::rowSums(fcm) * ncol(fcm)
		col_sums <- rep(1, ncol(fcm))
	}else{
		stop("`prob` must be one of NULL, 'rows', or 'cols'")
	}

	if (context_smoothing != 1) {
		col_sums <- col_sums^context_smoothing
	}
	if (target_smoothing != 1) {
		row_sums <- row_sums^target_smoothing
	}
	col_denom <- sum(col_sums)
	row_denom <- sum(row_sums)

	# PMI(w, c) = log( #(w,c) * col_denom * row_denom / (row_sums[w] * col_sums[c] * total_count) )
	if (is_quanteda || is_sparseMatrix || is_SparseArray) {
		numerator <- fcm@x * col_denom * row_denom
		denominator <- row_sums[fcm@i + 1] * col_sums[fcm@j + 1] * total_count
		fcm@x <- log(pmax(numerator, .Machine$double.eps), base) - log(pmax(denominator, .Machine$double.eps), base)

		if (shift != 0) {
			fcm@x <- fcm@x + shift
		}

		if (positive) {
			fcm@x <- pmax(fcm@x, 0)
			fcm <- Matrix::drop0(fcm)
		}else{
			warning("`positive = FALSE` sets all fcm zeros to -Inf. Coercing to dense matrix.")
			nz_indices <- (fcm@j * nrow(fcm)) + (fcm@i + 1)
			fcm <- as.matrix(fcm)
			fcm[-nz_indices] <- -Inf
		}
	} else {
		zero_indices <- fcm == 0
		numerator <- fcm * col_denom * row_denom
		denominator <- outer(row_sums, col_sums, `*`) * total_count
		fcm <- log(pmax(numerator, .Machine$double.eps), base) - log(pmax(denominator, .Machine$double.eps), base)

		if (shift != 0) {
			fcm <- fcm + shift
		}

		if (positive) {
			fcm <- pmax(fcm, 0)
		}else{
			fcm[zero_indices] <- -Inf
		}
	}

	# convert back to proper format
	if (is_quanteda) {
		fcm <- quanteda::as.fcm(fcm)
		fcm@meta <- fcm_meta
		fcm@meta$object$count <- "weighted"
	}
	if (is_sparseMatrix && positive) {
		fcm <- as(fcm, "CsparseMatrix")
	}
	if (is_SparseArray && positive) {
		fcm <- as(fcm, "SparseArray")
	}
	fcm
}

#' @rdname fcm_transformations
#' @param method smoothing method:
#' \describe{
#'   \item{`goodturing`}{the "Simple Good-Turing" algorithm described by Gale &
#'   Sampson (1995)}
#'   \item{`laplace`}{Laplace (a.k.a. "add one") smoothing}
#' }
#' @param crit criterion for switching between raw and smoothed estimates when
#'	`method = "goodturing"`. The default `1.96` corresponds to the standard
#'	0.05 significance criterion.
#' @param estimate_zeros logical; if `TRUE`, distribute the estimated
#'	probability of unobserved tokens among features with a count of zero. 
#'  Note that this coerces the output to a dense matrix, so may exceed memory 
#'  requirements for large FCMs.
#'
#' @export
fcm_smooth <- function(fcm, method = "goodturing", crit = 1.96, estimate_zeros = TRUE) {
	if( is_quanteda <- inherits(fcm, "fcm") ) {
		fcm_meta <- fcm@meta
	}
	is_sparseMatrix <- inherits(fcm, "sparseMatrix")
	is_SparseArray <- inherits(fcm, "SparseArray")
	stopifnot(
		"Unsupported fcm type: must be Quanteda fcm, sparseMatrix, matrix, 2D/3D array, or 2D/3D SparseArray" = (is_sparseMatrix || is_quanteda || is_SparseArray || is.array(fcm)) && (length(dim(fcm)) %in% c(2,3))
	)
	if (length(dim(fcm)) == 3) {
		fcm_ids <- dimnames(fcm)[[3]]
		fcm <- lapply(seq_len(dim(fcm)[3]), function(i) {
			fcm_smooth(fcm[,,i], method = method, crit = crit, estimate_zeros = estimate_zeros)
		})
		fcm <- do.call(S4Arrays::abind, c(fcm, along = 3))
		dimnames(fcm)[[3]] <- fcm_ids
		return(fcm)
	}

	# convert to TsparseMatrix if necessary
	if (is_quanteda || is_sparseMatrix || is_SparseArray) {
		fcm <- methods::as(fcm, "TsparseMatrix")
	}else{
		fcm <- as.matrix(fcm)
	}

	if (method == "goodturing") {
		if (is_quanteda || is_sparseMatrix || is_SparseArray) {
			n_doc <- table(fcm@x)
		}else{
			n_doc <- table(fcm[fcm != 0])
		}
		r <- as.numeric(as.character(names(n_doc)))
		N <- sum(n_doc * r)
		names(r) <- r
		if (length(n_doc) <= 1 || min(r) > 1){
			warning('FCM has only one unique term frequency, or does not contain hapax legomena. Returning raw counts.')
			P0 <- 0
			p <- r
		}else{
			Z <- 2 * n_doc / diff(c(0, r, 2 * r[length(r)] - r[length(r)-1]), lag = 2)

			lmfit <- stats::lm(log(Z) ~ log(r))

			r_star <- rep(NA, length(r))
			r_star_x <- rep(NA, length(r))
			r_star_y <- r * (1 + 1/r)^(lmfit$coefficients[2] + 1)

			crit_passed <- FALSE
			for (j in 1:length(n_doc)) {
				# check for gap in Nr
				if (!crit_passed) {
					if (is.na(n_doc[as.character(r[j] + 1)])) {
						crit_passed <- TRUE
					}
				}
				# compute r_star
				if (!crit_passed) {
					r_star_x[j] <- (r[j] + 1) * n_doc[j + 1] / n_doc[j]
					crit_passed <- ((r_star_x[j] - r_star_y[j])/crit)^2 < (1 + n_doc[j + 1] / n_doc[j]) * ((r[j] + 1)^2) * (n_doc[j + 1] / n_doc[j]^2)
					r_star[j] <- ifelse(crit_passed, r_star_y[j], r_star_x[j])
				} else {
					r_star[j] <- r_star_y[j]
				}
			}
			names(r_star) <- r
			Nprime <- sum(n_doc * r_star)
			P0 <- n_doc[1]
			p <- (1 - (P0 / N)) * (r_star / Nprime)
			p <- p * N
		}

		if (is_quanteda || is_sparseMatrix || is_SparseArray) {
			fcm@x <- p[as.character(fcm@x)]
		}else{
			fcm[fcm != 0] <- p[as.character(fcm[fcm != 0])]
		}

		if (estimate_zeros){
			if (is_quanteda || is_sparseMatrix || is_SparseArray) {
				nz_indices <- (fcm@j * nrow(fcm)) + (fcm@i + 1)
				fcm <- as.matrix(fcm)
				fcm[-nz_indices] <- P0/(length(fcm) - length(nz_indices))
			}else{
				nz_indices <- which(fcm != 0, arr.ind = TRUE)
				fcm[fcm == 0] <- P0/(length(fcm) - length(nz_indices))
			}
		}
	}else if (method == "laplace") {
		if (is_quanteda || is_sparseMatrix || is_SparseArray) {
			fcm@x <- fcm@x + 1
			if (estimate_zeros) {
				nz_indices <- (fcm@j * nrow(fcm)) + (fcm@i + 1)
				fcm <- as.matrix(fcm)
				fcm[-nz_indices] <- 1
			}
		}else{
			if (estimate_zeros) {
				fcm <- fcm + 1
			}else{
				fcm[fcm != 0] <- fcm[fcm != 0] + 1
			}
		}
	}

	# convert back to proper format
	if (is_quanteda) {
		fcm <- quanteda::as.fcm(fcm)
		fcm@meta <- fcm_meta
		fcm@meta$object$count <- "weighted"
	}
	if (is_sparseMatrix && !estimate_zeros) {
		fcm <- as(fcm, "CsparseMatrix")
	}
	if (is_SparseArray && !estimate_zeros) {
		fcm <- as(fcm, "SparseArray")
	}
	fcm
}

#' @rdname fcm_transformations
#' @export
fcm_positive <- function(fcm) {
	if (inherits(fcm, "Matrix") || quanteda::is.fcm(fcm)) {
		if (any(fcm@x <= 0)) {
			fcm@x[fcm@x <= 0] <- 0
			if (quanteda::is.fcm(fcm)) {
				attrs <- attributes(fcm)
				fcm <- quanteda::as.fcm(Matrix::drop0(fcm))
				fcm@meta <- attrs[["meta"]]
				fcm@meta$object$count <- "weighted"
			}else{
				fcm <- Matrix::drop0(fcm)
			}
		}
	}else if (inherits(fcm, c("SparseArray"))) {
		SparseArray::nzvals(fcm)[SparseArray::nzvals(fcm) < 0] <- 0
	}else{
		fcm[fcm < 0] <- 0
	}
	fcm
}

#' @rdname fcm_transformations
#' @export
fcm_log <- function(fcm, positive = TRUE, base = 2) {
	if (!positive) {
		if (length(dim(fcm)) == 2) {
			fcm <- as.matrix(fcm)
		}else{
			fcm <- as.array(fcm)
		}
		return(log(fcm, base = base))
	}
	if (inherits(fcm, "Matrix") || quanteda::is.fcm(fcm)) {
		if (any(fcm@x <= 1)) {
			fcm@x[fcm@x < 1] <- 1
		}
		fcm@x <- log(fcm@x, base = base)
		if (quanteda::is.fcm(fcm)) {
			attrs <- attributes(fcm)
			fcm <- quanteda::as.fcm(Matrix::drop0(fcm))
			fcm@meta <- attrs[["meta"]]
			fcm@meta$object$count <- "weighted"
		}else{
			fcm <- Matrix::drop0(fcm)
		}
	}else if (inherits(fcm, c("SparseArray"))) {
		SparseArray::nzvals(fcm)[SparseArray::nzvals(fcm) < 1] <- 1
		SparseArray::nzvals(fcm) <- log(SparseArray::nzvals(fcm), base = base)
	}else{
		fcm[fcm < 1] <- 1
		fcm <- log(fcm, base = base)
	}
	fcm
}
