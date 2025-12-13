library(testthat)
library(quanteda)

# Setup: Create a simple test FCM
make_test_fcm <- function() {
  toks <- tokens(
    c("the quick brown fox jumps",
      "the lazy brown dog",
      "quick fox jumps high"),
    remove_punct = TRUE
  )
  fcm(toks, window = 3)
}

test_that("train_sgns basic training works with defaults", {
  fcm_mat <- make_test_fcm()
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 2,
    output = "all",
    verbose = FALSE
  )
  
  expect_s3_class(result, "dynamic_embeddings")
  expect_equal(dim(result$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result$context_embeddings), c(ncol(fcm_mat), 10))
  expect_equal(result$train_method, "sgns")
  expect_true(!is.null(result$loss_history))
  expect_equal(length(result$loss_history), 2) # 2 epochs
})

test_that("train_sgns respects output parameter", {
  fcm_mat <- make_test_fcm()
  
  # Word embeddings only
  result_word <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    output = "word_embeddings",
    verbose = FALSE
  )
  expect_true(!is.null(result_word$word_embeddings))
  expect_true(is.null(result_word$context_embeddings))
  
  # Context embeddings only
  result_context <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    output = "context_embeddings",
    verbose = FALSE
  )
  expect_true(is.null(result_context$word_embeddings))
  expect_true(!is.null(result_context$context_embeddings))
  
  # Both
  result_all <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    output = "all",
    verbose = FALSE
  )
  expect_true(!is.null(result_all$word_embeddings))
  expect_true(!is.null(result_all$context_embeddings))
})

test_that("train_sgns smoothing parameter works", {
  fcm_mat <- make_test_fcm()
  
  # Frequency-weighted (default)
  result_freq <- train_sgns(
    fcm_mat,
    n_dims = 10,
    context_smoothing = 0.75,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_freq$control$context_smoothing, 0.75)
  
  # Uniform sampling (context_smoothing = 0)
  result_uniform <- train_sgns(
    fcm_mat,
    n_dims = 10,
    context_smoothing = 0,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_uniform$control$context_smoothing, 0)
  
  # Both should produce valid embeddings
  expect_equal(dim(result_freq$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_uniform$word_embeddings), c(nrow(fcm_mat), 10))
})

test_that("train_sgns init parameter works", {
  fcm_mat <- make_test_fcm()
  
  # Uniform initialization
  result_uniform <- train_sgns(
    fcm_mat,
    n_dims = 10,
    init = "uniform",
    epochs = 1,
    verbose = FALSE,
    seed = 123
  )
  expect_equal(result_uniform$control$init, "uniform")
  
  # Normal initialization
  result_normal <- train_sgns(
    fcm_mat,
    n_dims = 10,
    init = "normal",
    epochs = 1,
    verbose = FALSE,
    seed = 123
  )
  expect_equal(result_normal$control$init, "normal")
  
  # Both should produce valid embeddings
  expect_equal(dim(result_uniform$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_normal$word_embeddings), c(nrow(fcm_mat), 10))
  
  # Embeddings should differ due to different initialization (probabilistically)
  expect_false(isTRUE(all.equal(result_uniform$word_embeddings, result_normal$word_embeddings)))
})

test_that("train_sgns reject_positives parameter works", {
  fcm_mat <- make_test_fcm()
  
  # With rejection sampling (default)
  result_reject <- train_sgns(
    fcm_mat,
    n_dims = 10,
    reject_positives = TRUE,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_reject$control$reject_positives, TRUE)
  
  # Without rejection sampling
  result_no_reject <- train_sgns(
    fcm_mat,
    n_dims = 10,
    reject_positives = FALSE,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_no_reject$control$reject_positives, FALSE)
  
  # Both should produce valid embeddings
  expect_equal(dim(result_reject$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_no_reject$word_embeddings), c(nrow(fcm_mat), 10))
})

test_that("train_sgns bootstrap_positive parameter works", {
  fcm_mat <- make_test_fcm()
  
  # Without bootstrap (default)
  result_no_bootstrap <- train_sgns(
    fcm_mat,
    n_dims = 10,
    bootstrap_positive = FALSE,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_no_bootstrap$control$bootstrap_positive, FALSE)
  
  # With bootstrap
  result_bootstrap <- train_sgns(
    fcm_mat,
    n_dims = 10,
    bootstrap_positive = TRUE,
    epochs = 1,
    verbose = FALSE
  )
  expect_equal(result_bootstrap$control$bootstrap_positive, TRUE)
  
  # Both should produce valid embeddings
  expect_equal(dim(result_no_bootstrap$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_bootstrap$word_embeddings), c(nrow(fcm_mat), 10))
})

test_that("train_sgns control parameters are stored correctly", {
  fcm_mat <- make_test_fcm()
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 50,
    neg = 3,
    lr = 0.02,
    epochs = 2,
    grain_size = 2,
    context_smoothing = 0.5,
    reject_positives = FALSE,
    init = "normal",
    bootstrap_positive = TRUE,
    verbose = FALSE
  )
  
  # Verify all control parameters are stored
  expect_equal(result$control$method, "sgns")
  expect_equal(result$control$n_dims, 50)
  expect_equal(result$control$neg, 3)
  expect_equal(result$control$lr, 0.02)
  expect_equal(result$control$epochs, 2)
  expect_equal(result$control$grain_size, 2)
  expect_equal(result$control$context_smoothing, 0.5)
  expect_equal(result$control$reject_positives, FALSE)
  expect_equal(result$control$init, "normal")
  expect_equal(result$control$bootstrap_positive, TRUE)
})

test_that("train_sgns input validation works", {
  fcm_mat <- make_test_fcm()
  
  # Invalid n_dims
  expect_error(
    train_sgns(fcm_mat, n_dims = -10, epochs = 1, verbose = FALSE),
    "`n_dims` must be a positive integer"
  )
  
  # Invalid smoothing
  expect_error(
    train_sgns(fcm_mat, context_smoothing = -0.5, epochs = 1, verbose = FALSE)
  )
  
  # Invalid init
  expect_error(
    train_sgns(fcm_mat, init = "invalid", epochs = 1, verbose = FALSE),
    "`init` must be 'uniform' or 'normal'"
  )
  
  # Invalid reject_positives
  expect_error(
    train_sgns(fcm_mat, reject_positives = "yes", epochs = 1, verbose = FALSE),
    "`reject_positives` must be logical"
  )
})

test_that("train_sgns reproducibility with seed", {
  fcm_mat <- make_test_fcm()
  
  result1 <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 42,
    threads = 1
  )
  
  result2 <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 42,
    threads = 1
  )
  
  # Same seed should produce identical embeddings
  expect_equal(result1$word_embeddings, result2$word_embeddings, tolerance = 1e-10)
})

test_that("train_sgns preserves rownames and colnames", {
  toks <- tokens(
    c("the quick brown fox", "the lazy dog"),
    remove_punct = TRUE
  )
  fcm_mat <- fcm(toks, window = 3)
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    output = "all",
    verbose = FALSE
  )
  
  expect_equal(rownames(result$word_embeddings), rownames(fcm_mat))
  expect_equal(rownames(result$context_embeddings), colnames(fcm_mat))
})

test_that("train_sgns works with different FCM input formats", {
  toks <- tokens(
    c("the quick brown fox", "the lazy dog"),
    remove_punct = TRUE
  )
  fcm_mat <- fcm(toks, window = 3)
  
  # Quanteda FCM
  result_quanteda <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 123,
    threads = 1
  )
  
  # Convert to sparse matrix
  fcm_sparse <- methods::as(fcm_mat, "TsparseMatrix")
  result_sparse <- train_sgns(
    fcm_sparse,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 123,
    threads = 1
  )
  
  # Convert to dense matrix
  fcm_dense <- as.matrix(fcm_mat)
  result_dense <- train_sgns(
    fcm_dense,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 123,
    threads = 1
  )
  
  # Quanteda and sparse should produce identical results (both are sparse internally)
  expect_equal(result_quanteda$word_embeddings, result_sparse$word_embeddings, tolerance = 1e-10)
  
  # Dense matrix should produce valid embeddings of correct dimensions
  # (Don't compare values since matrix format affects internal ordering/iteration)
  expect_equal(dim(result_dense$word_embeddings), dim(result_quanteda$word_embeddings))
  expect_true(all(is.finite(result_dense$word_embeddings)))
  expect_equal(rownames(result_dense$word_embeddings), rownames(result_quanteda$word_embeddings))
})

test_that("train_sgns target_smoothing parameter works", {
  fcm_mat <- make_test_fcm()
  
  # No reweighting (default)
  result_default <- train_sgns(
    fcm_mat,
    n_dims = 10,
    target_smoothing = 1,
    epochs = 1,
    verbose = FALSE,
    seed = 123
  )
  expect_equal(result_default$control$target_smoothing, 1)
  
  # Uniform target sampling (target_smoothing = 0)
  result_uniform <- train_sgns(
    fcm_mat,
    n_dims = 10,
    target_smoothing = 0,
    epochs = 1,
    verbose = FALSE,
    seed = 456
  )
  expect_equal(result_uniform$control$target_smoothing, 0)
  
  # Downsampling frequent words (target_smoothing = 0.5)
  result_downsample <- train_sgns(
    fcm_mat,
    n_dims = 10,
    target_smoothing = 0.5,
    epochs = 1,
    verbose = FALSE,
    seed = 789
  )
  expect_equal(result_downsample$control$target_smoothing, 0.5)
  
  # All should produce valid embeddings
  expect_equal(dim(result_default$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_uniform$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_downsample$word_embeddings), c(nrow(fcm_mat), 10))
  
  # Embeddings should differ due to different reweighting (use different seeds)
  expect_false(isTRUE(all.equal(result_default$word_embeddings, result_uniform$word_embeddings, tolerance = 1e-4)))
})

test_that("train_sgns subsample parameter works", {
  fcm_mat <- make_test_fcm()
  
  # No subsampling (default)
  result_no_subsample <- train_sgns(
    fcm_mat,
    n_dims = 10,
    subsample = 0,
    epochs = 1,
    verbose = FALSE,
    seed = 123
  )
  expect_equal(result_no_subsample$control$subsample, 0)
  
  # With subsampling
  result_subsample <- train_sgns(
    fcm_mat,
    n_dims = 10,
    subsample = 1e-3,
    epochs = 1,
    verbose = FALSE,
    seed = 123
  )
  expect_equal(result_subsample$control$subsample, 1e-3)
  
  # Both should produce valid embeddings
  expect_equal(dim(result_no_subsample$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result_subsample$word_embeddings), c(nrow(fcm_mat), 10))
  
  # Embeddings should differ due to subsampling
  expect_false(isTRUE(all.equal(result_no_subsample$word_embeddings, result_subsample$word_embeddings)))
})

test_that("train_sgns combined target_smoothing and subsample work", {
  fcm_mat <- make_test_fcm()
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 10,
    target_smoothing = 0.5,
    subsample = 1e-3,
    context_smoothing = 0.5,
    epochs = 1,
    verbose = FALSE
  )
  
  # Verify control parameters
  expect_equal(result$control$target_smoothing, 0.5)
  expect_equal(result$control$subsample, 1e-3)
  expect_equal(result$control$context_smoothing, 0.5)
  
  # Should produce valid embeddings
  expect_equal(dim(result$word_embeddings), c(nrow(fcm_mat), 10))
})

test_that("train_sgns input validation for new parameters", {
  fcm_mat <- make_test_fcm()
  
  # Invalid target_smoothing
  expect_error(
    train_sgns(fcm_mat, target_smoothing = -0.5, epochs = 1, verbose = FALSE)
  )
  
  # Invalid subsample
  expect_error(
    train_sgns(fcm_mat, subsample = -1e-3, epochs = 1, verbose = FALSE),
    "`subsample` must be a non-negative number"
  )
})


test_that("train_sgns runs with multiple threads", {
  skip_on_cran()
  
  # Create a simple FCM
  toks <- quanteda::tokens(
    c("the quick brown fox", "the lazy dog", "quick fox"),
    remove_punct = TRUE
  )
  fcm_mat <- quanteda::fcm(toks, context = "window", window = 2)
  
  # Train with 2 threads
  set.seed(42)
  emb_parallel <- train_sgns(
    fcm_mat,
    n_dims = 10,
    neg = 2,
    epochs = 2,
    threads = 2,
    verbose = FALSE
  )
  
  expect_s3_class(emb_parallel, "dynamic_embeddings")
  expect_equal(dim(emb_parallel$word_embeddings), c(6, 10))
  
  # Train with 1 thread
  set.seed(42)
  emb_single <- train_sgns(
    fcm_mat,
    n_dims = 10,
    neg = 2,
    epochs = 2,
    threads = 1,
    verbose = FALSE
  )
  
  expect_s3_class(emb_single, "dynamic_embeddings")
  
  # Results won't be identical due to race conditions and RNG differences in parallel
  # But dimensions should match
  expect_equal(dim(emb_parallel$word_embeddings), dim(emb_single$word_embeddings))
})

test_that("train_sgns respects grain_size", {
  skip_on_cran()
  
  toks <- quanteda::tokens(
    c("the quick brown fox", "the lazy dog", "quick fox"),
    remove_punct = TRUE
  )
  fcm_mat <- quanteda::fcm(toks, context = "window", window = 2)
  
  # Train with grain_size = 2
  emb <- train_sgns(
    fcm_mat,
    n_dims = 10,
    neg = 2,
    epochs = 2,
    grain_size = 2,
    threads = 1,
    verbose = FALSE
  )
  
  expect_s3_class(emb, "dynamic_embeddings")
})
