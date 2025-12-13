#' Specify Context Window Configuration
#'
#' Creates a context specification object that defines how context windows are
#' constructed around target words. This specification can be reused across
#' different embedding methods (FCM construction, SGNS training, etc.) to ensure
#' consistent treatment of context.
#'
#' @param window Size of the context window (in words) on either side of the target word.
#' @param weights Either a string specifying a decay function of distance 
#'   ("linear", "harmonic", "exponential", "power", or "none") or a numeric 
#'   vector of weights. If a vector, its length must be equal to `window` 
#'   (if `include_target = FALSE`) or `window + 1` (if `include_target = TRUE`) 
#'   for one-sided context, or `2 * window` / `2 * window + 1` for two-sided context.
#' @param weights_args List of arguments for the decay function (e.g. `alpha` for power/exp).
#' @param distance_metric Metric to input to decay function: "words", "characters", 
#'   "surprisal". "words" uses token distance; "characters" weights words by their 
#'   length in characters; "surprisal" weights words by -log(probability) based on 
#'   unigram frequencies. Metrics are rescaled such that the average width of a 
#'   word is fixed at 1 (i.e. `window = 5` will result in an average window size 
#'   of 5 words).
#' @param direction String ("symmetric", "forward", "backward") or numeric ratio 
#'   of forward to backward weight.
#' @param include_target Logical. If TRUE, the target word is included in the 
#'   context (distance 0).
#'
#' @return A `context_spec` object containing the context configuration.
#'
#' @export
#' @examples
#' # Standard symmetric window
#' ctx <- context_spec(window = 5, weights = "linear")
#' 
#' # Asymmetric window favoring forward context
#' ctx <- context_spec(window = 5, direction = 2.0)  # 2:1 forward:backward
#' 
#' # Custom weight vector
#' ctx <- context_spec(window = 3, weights = c(0.5, 0.8, 1.0))
#' 
#' # Use with FCM
#' fcm_mat <- fcm(tokens, context = ctx)
#' 
#' # Use with SGNS (streaming mode)
#' model <- train_sgns(tokens, context = ctx, n_dims = 100)
context_spec <- function(
  window = 5L,
  weights = "linear",
  weights_args = list(),
  distance_metric = c("words", "characters", "surprisal"),
  direction = "symmetric",
  include_target = FALSE
) {
  
  distance_metric <- match.arg(distance_metric)
  
  # Validate weights
  if (is.character(weights)) {
    weights <- match.arg(weights, c("linear", "harmonic", "exponential", "power", "none"))
  } else if (is.numeric(weights)) {
    # Will validate length when actually used (depends on include_target)
    if (!is.numeric(weights) || length(weights) < 1) {
      stop("weights must be a non-empty numeric vector or decay function name")
    }
  } else {
    stop("weights must be a numeric vector or a decay function name")
  }
  
  # Validate direction
  if (is.numeric(direction)) {
    if (direction < 0) {
      stop("direction ratio must be non-negative")
    }
  } else if (is.character(direction)) {
    direction <- match.arg(direction, c("symmetric", "forward", "backward"))
  } else {
    stop('direction must be "symmetric", "forward", "backward", or a numeric ratio')
  }
  
  structure(
    list(
      window = as.integer(window),
      weights = weights,
      weights_args = weights_args,
      distance_metric = distance_metric,
      direction = direction,
      include_target = as.logical(include_target)
    ),
    class = "context_spec"
  )
}

#' @export
print.context_spec <- function(x, ...) {
  cat("Context Specification:\n")
  cat(sprintf("  Window: %d\n", x$window))
  cat(sprintf("  Weights: %s\n", 
              if(is.character(x$weights)) x$weights else "custom vector"))
  cat(sprintf("  Direction: %s\n", 
              if(is.character(x$direction)) x$direction else sprintf("%.2f:1", x$direction)))
  cat(sprintf("  Distance metric: %s\n", x$distance_metric))
  cat(sprintf("  Include target: %s\n", x$include_target))
  invisible(x)
}
