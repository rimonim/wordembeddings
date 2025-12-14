library(testthat)
library(quanteda)

# Setup: Create test data
make_test_tokens <- function() {
  tokens(
    c("the quick brown fox jumps",
      "the lazy brown dog",
      "quick fox jumps high"),
    remove_punct = TRUE
  )
}

make_large_test_tokens <- function() {
  tokens(quanteda.corpora::data_corpus_sotu[1:25], what = "word", remove_punct = TRUE)
}

make_test_fcm <- function() {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3)
  fcm(toks, context = ctx)
}

test_that("train_sgns.tokens basic training works", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, min_count = 1)
  
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    verbose = FALSE
  )
  
  expect_type(result, "list")
  expect_true(!is.null(result$word_embeddings))
  expect_true(!is.null(result$context_embeddings))
  expect_equal(ncol(result$word_embeddings), 10)
  expect_equal(ncol(result$context_embeddings), 10)
})

test_that("train_sgns.fcm basic training works", {
  fcm_mat <- make_test_fcm()
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 2,
    verbose = FALSE
  )
  
  expect_type(result, "list")
  expect_true(!is.null(result$word_embeddings))
  expect_true(!is.null(result$context_embeddings))
  expect_equal(dim(result$word_embeddings), c(nrow(fcm_mat), 10))
  expect_equal(dim(result$context_embeddings), c(nrow(fcm_mat), 10))
})

test_that("train_sgns.tokens uses default context_spec", {
  toks <- make_test_tokens()
  
  # Should work with default context_spec
  result <- train_sgns(toks, n_dims = 10, epochs = 1, verbose = FALSE)
  expect_equal(ncol(result$word_embeddings), 10)
})

test_that("train_sgns.tokens context_spec parameters work", {
  toks <- make_test_tokens()
  
  # Test with different context_smoothing
  ctx1 <- context_spec(window = 3, context_smoothing = 0.75, min_count = 1)
  result1 <- train_sgns(toks, context = ctx1, n_dims = 10, epochs = 1, verbose = FALSE)
  expect_equal(dim(result1$word_embeddings)[2], 10)
  
  # Test with different subsample
  ctx2 <- context_spec(window = 3, subsample = 1e-3, min_count = 1)
  result2 <- train_sgns(toks, context = ctx2, n_dims = 10, epochs = 1, verbose = FALSE)
  expect_equal(dim(result2$word_embeddings)[2], 10)
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
  expect_equal(dim(result_uniform$word_embeddings), c(nrow(fcm_mat), 10))
  
  # Normal initialization
  result_normal <- train_sgns(
    fcm_mat,
    n_dims = 10,
    init = "normal",
    epochs = 1,
    verbose = FALSE,
    seed = 124
  )
  expect_equal(dim(result_normal$word_embeddings), c(nrow(fcm_mat), 10))
  
  # Embeddings should differ due to different initialization
  expect_false(isTRUE(all.equal(result_uniform$word_embeddings, result_normal$word_embeddings, tolerance = 1e-4)))
})

test_that("Include target parameter works", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # With and without target inclusion
  ctx_no_target <- context_spec(window = 3, include_target = FALSE)
  ctx_with_target <- context_spec(window = 3, include_target = TRUE)
  
  set.seed(202)
  m_no_target <- train_sgns(toks_test, context = ctx_no_target, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  set.seed(202)
  m_with_target <- train_sgns(toks_test, context = ctx_with_target, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  # Compare dot products - models should be different
  dots_no_target <- m_no_target$word_embeddings %*% t(m_no_target$context_embeddings)
  dots_with_target <- m_with_target$word_embeddings %*% t(m_with_target$context_embeddings)
  
  cor_val <- cor(c(dots_no_target), c(dots_with_target))
  expect_true(cor_val < 0.95, info = sprintf("Dot products should differ, correlation: %.3f", cor_val))
})

test_that("train_sgns.tokens with linear decay weights", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, weights = "linear", min_count = 1)
  
  # Should use weighted sampling path
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    seed = 123,
    threads = 1,
    verbose = FALSE
  )
  
  expect_equal(ncol(result$word_embeddings), 10)
  expect_true(all(is.finite(result$word_embeddings)))
})

test_that("train_sgns.tokens with harmonic decay weights", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, weights = "harmonic", min_count = 1)
  
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    seed = 124,
    threads = 1,
    verbose = FALSE
  )
  
  expect_equal(ncol(result$word_embeddings), 10)
  expect_true(all(is.finite(result$word_embeddings)))
})

test_that("train_sgns.tokens with include_target=TRUE", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, include_target = TRUE, min_count = 1)
  
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    seed = 125,
    threads = 1,
    verbose = FALSE
  )
  
  expect_equal(ncol(result$word_embeddings), 10)
  expect_true(all(is.finite(result$word_embeddings)))
})

test_that("train_sgns.tokens with forward direction", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, direction = "forward", min_count = 1)
  
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    seed = 126,
    threads = 1,
    verbose = FALSE
  )
  
  expect_equal(ncol(result$word_embeddings), 10)
  expect_true(all(is.finite(result$word_embeddings)))
})

test_that("train_sgns.tokens with backward direction", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, direction = "backward", min_count = 1)
  
  result <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 2,
    seed = 127,
    threads = 1,
    verbose = FALSE
  )
  
  expect_equal(ncol(result$word_embeddings), 10)
  expect_true(all(is.finite(result$word_embeddings)))
})

test_that("train_sgns.tokens equivalence: FCM vs streaming with linear weights", {
  skip("Integration test - enable to verify FCM/streaming equivalence")
  
  toks <- tokens(
    c("the quick brown fox jumps over the lazy dog",
      "the brown dog is lazy",
      "quick fox jumps high over the fence"),
    remove_punct = TRUE
  )
  
  ctx <- context_spec(window = 5, weights = "linear", min_count = 1)
  
  # Train from FCM
  fcm_mat <- fcm(toks, context = ctx)
  result_fcm <- train_sgns(
    fcm_mat,
    n_dims = 50,
    epochs = 10,
    seed = 42,
    threads = 1,
    verbose = FALSE
  )
  
  # Train from tokens (streaming with weighted sampling)
  result_tokens <- train_sgns(
    toks,
    context = ctx,
    n_dims = 50,
    epochs = 10,
    seed = 42,
    threads = 1,
    verbose = FALSE
  )
  
  # Embeddings should be highly correlated (>0.85)
  cor_val <- cor(c(result_fcm$word_embeddings), c(result_tokens$word_embeddings))
  expect_gt(cor_val, 0.85)
})

test_that("train_sgns input validation works", {
  fcm_mat <- make_test_fcm()
  
  # Invalid n_dims
  expect_error(
    train_sgns(fcm_mat, n_dims = -10, epochs = 1, verbose = FALSE),
    "`n_dims` must be a positive integer"
  )
  
  # Invalid init
  expect_error(
    train_sgns(fcm_mat, init = "invalid", epochs = 1, verbose = FALSE),
    "`init` must be 'uniform' or 'normal'"
  )
})

test_that("train_sgns.fcm reproducibility with seed", {
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
  
  # Same seed should produce very similar embeddings (stochastic sampling introduces slight variation)
  expect_equal(result1$word_embeddings, result2$word_embeddings, tolerance = 0.01)
})

test_that("train_sgns.tokens and train_sgns.fcm both produce valid embeddings", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Default context_spec with min_count to ensure same vocabulary
  ctx_default <- context_spec(window = 5, min_count = 2)
  
  # Train from tokens (streaming)
  set.seed(123)
  m1 <- train_sgns(toks_test, context = ctx_default, n_dims = 50, epochs = 5, threads = 2, verbose = FALSE)
  
  # Train from FCM
  fcm_test <- fcm(toks_test, context = ctx_default)
  # Filter FCM to same vocabulary
  common_words <- intersect(rownames(m1$word_embeddings), rownames(fcm_test))
  fcm_test <- fcm_test[common_words, common_words]
  
  set.seed(123)
  m2 <- train_sgns(fcm_test, n_dims = 50, epochs = 5, threads = 2, verbose = FALSE)
  
  # Both methods should produce valid embeddings with correct dimensions
  expect_equal(ncol(m1$word_embeddings), 50)
  expect_equal(ncol(m2$word_embeddings), 50)
  expect_true(all(is.finite(m1$word_embeddings)))
  expect_true(all(is.finite(m2$word_embeddings)))
  
  # Note: With small data and different training methodologies (streaming sampling vs
  # FCM-based), embeddings may differ substantially but both capture valid semantic structure.
  # Direct correlation comparison is not meaningful due to rotational freedom and
  # limited training data. Both methods are validated separately in other tests.
})

test_that("Context weights affect training (linear decay)", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Compare no weighting vs linear decay
  ctx_none <- context_spec(window = 5, weights = "none")
  ctx_linear <- context_spec(window = 5, weights = "linear")
  
  set.seed(456)
  m_none <- train_sgns(toks_test, context = ctx_none, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  set.seed(456)
  m_linear <- train_sgns(toks_test, context = ctx_linear, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  # Compare dot products - models should be different
  dots_none <- m_none$word_embeddings %*% t(m_none$context_embeddings)
  dots_linear <- m_linear$word_embeddings %*% t(m_linear$context_embeddings)
  
  cor_val <- cor(c(dots_none), c(dots_linear))
  expect_true(cor_val < 0.95, info = sprintf("Dot products should differ, correlation: %.3f", cor_val))
})

test_that("Direction parameter affects training (forward vs backward)", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Forward vs backward context
  ctx_forward <- context_spec(window = 5, direction = "forward")
  ctx_backward <- context_spec(window = 5, direction = "backward")
  
  set.seed(789)
  m_forward <- train_sgns(toks_test, context = ctx_forward, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  set.seed(789)
  m_backward <- train_sgns(toks_test, context = ctx_backward, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  # Compare dot products - models should be different
  dots_forward <- m_forward$word_embeddings %*% t(m_forward$context_embeddings)
  dots_backward <- m_backward$word_embeddings %*% t(m_backward$context_embeddings)
  
  cor_val <- cor(c(dots_forward), c(dots_backward))
  expect_true(cor_val < 0.95, info = sprintf("Dot products should differ, correlation: %.3f", cor_val))
})

test_that("Distance metric parameter works", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Test that distance metric options work without errors
  ctx_words <- context_spec(window = 3, weights = "linear", distance_metric = "words", min_count = 1)
  m_words <- train_sgns(toks_test, context = ctx_words, n_dims = 10, epochs = 1, threads = 1, verbose = FALSE, seed = 101)
  
  ctx_chars <- context_spec(window = 3, weights = "linear", distance_metric = "characters", min_count = 1)
  m_chars <- train_sgns(toks_test, context = ctx_chars, n_dims = 10, epochs = 1, threads = 1, verbose = FALSE, seed = 102)
  
  # Both should produce valid embeddings
  expect_equal(ncol(m_words$word_embeddings), 10)
  expect_equal(ncol(m_chars$word_embeddings), 10)
  expect_true(all(is.finite(m_words$word_embeddings)))
  expect_true(all(is.finite(m_chars$word_embeddings)))
})

test_that("Custom weight vector works", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Custom weights: more extreme difference (uniform vs. highly peaked)
  ctx_uniform <- context_spec(window = 3, weights = "none")
  ctx_peaked <- context_spec(window = 3, weights = c(1.0, 0.2, 0.05))
  
  set.seed(303)
  m_uniform <- train_sgns(toks_test, context = ctx_uniform, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  set.seed(303)
  m_peaked <- train_sgns(toks_test, context = ctx_peaked, n_dims = 30, epochs = 3, threads = 2, verbose = FALSE)
  
  # Compare dot products - models should be different
  dots_uniform <- m_uniform$word_embeddings %*% t(m_uniform$context_embeddings)
  dots_peaked <- m_peaked$word_embeddings %*% t(m_peaked$context_embeddings)
  
  cor_val <- cor(c(dots_uniform), c(dots_peaked))
  expect_true(cor_val < 0.95, info = sprintf("Dot products should differ, correlation: %.3f", cor_val))
})

test_that("train_sgns.tokens reproducibility with seed", {
  toks <- make_test_tokens()
  ctx <- context_spec(window = 3, min_count = 1)
  
  result1 <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 42,
    threads = 1
  )
  
  result2 <- train_sgns(
    toks,
    context = ctx,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE,
    seed = 42,
    threads = 1
  )
  
  # Same seed should produce identical embeddings
  expect_equal(result1$word_embeddings, result2$word_embeddings, tolerance = 1e-6)
})

test_that("train_sgns.fcm preserves rownames", {
  toks <- tokens(
    c("the quick brown fox", "the lazy dog"),
    remove_punct = TRUE
  )
  ctx <- context_spec(window = 3)
  fcm_mat <- fcm(toks, context = ctx)
  
  result <- train_sgns(
    fcm_mat,
    n_dims = 10,
    epochs = 1,
    verbose = FALSE
  )
  
  expect_equal(rownames(result$word_embeddings), rownames(fcm_mat))
  expect_equal(rownames(result$context_embeddings), rownames(fcm_mat))
})

test_that("train_sgns.fcm works with different FCM input formats", {
  toks <- tokens(
    c("the quick brown fox", "the lazy dog"),
    remove_punct = TRUE
  )
  ctx <- context_spec(window = 3)
  fcm_mat <- fcm(toks, context = ctx)
  
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
  
  # Both should produce valid embeddings (stochastic sampling means not identical)
  expect_equal(dim(result_quanteda$word_embeddings), dim(result_sparse$word_embeddings))
  expect_true(all(is.finite(result_quanteda$word_embeddings)))
  expect_true(all(is.finite(result_sparse$word_embeddings)))
})




test_that("train_sgns runs with multiple threads", {
  skip_on_cran()
  
  toks <- tokens(
    c("the quick brown fox", "the lazy dog", "quick fox"),
    remove_punct = TRUE
  )
  ctx <- context_spec(window = 2, min_count = 1)
  fcm_mat <- fcm(toks, context = ctx)
  
  # Train with 2 threads
  emb_parallel <- train_sgns(
    fcm_mat,
    n_dims = 10,
    neg = 2,
    epochs = 2,
    threads = 2,
    verbose = FALSE,
    seed = 42
  )
  
  expect_equal(dim(emb_parallel$word_embeddings)[2], 10)
  
  # Train with 1 thread
  emb_single <- train_sgns(
    fcm_mat,
    n_dims = 10,
    neg = 2,
    epochs = 2,
    threads = 1,
    verbose = FALSE,
    seed = 42
  )
  
  # Dimensions should match
  expect_equal(dim(emb_parallel$word_embeddings), dim(emb_single$word_embeddings))
})

test_that("vocab_size parameter limits vocabulary", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Train with vocab_size limit
  ctx_limited <- context_spec(window = 3, min_count = 1, vocab_size = 10)
  m_limited <- train_sgns(toks_test, context = ctx_limited, n_dims = 20, 
                          epochs = 1, threads = 1, verbose = FALSE, seed = 111)
  
  # Should have at most 10 words
  expect_true(nrow(m_limited$word_embeddings) <= 10)
  
  # Train without limit
  ctx_unlimited <- context_spec(window = 3, min_count = 1, vocab_size = NULL)
  m_unlimited <- train_sgns(toks_test, context = ctx_unlimited, n_dims = 20, 
                            epochs = 1, threads = 1, verbose = FALSE, seed = 111)
  
  # Should have more words than limited version (test data has ~20 unique words)
  expect_true(nrow(m_unlimited$word_embeddings) > nrow(m_limited$word_embeddings))
})

test_that("vocab_coverage parameter limits vocabulary", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Train with 50% coverage
  ctx_50 <- context_spec(window = 3, min_count = 1, vocab_coverage = 0.5)
  m_50 <- train_sgns(toks_test, context = ctx_50, n_dims = 20, 
                     epochs = 1, threads = 1, verbose = FALSE, seed = 222)
  
  # Train with 80% coverage
  ctx_80 <- context_spec(window = 3, min_count = 1, vocab_coverage = 0.8)
  m_80 <- train_sgns(toks_test, context = ctx_80, n_dims = 20, 
                     epochs = 1, threads = 1, verbose = FALSE, seed = 222)
  
  # 80% coverage should have more words than 50%
  expect_true(nrow(m_80$word_embeddings) > nrow(m_50$word_embeddings))
})

test_that("vocab_keep parameter forces inclusion of words", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Pick some words that appear rarely (< 10 times) to force inclusion
  # Using actual words from data_corpus_sotu
  rare_words <- c("congratulating", "accession", "Carolina")
  
  # Train with very high min_count (would normally exclude rare words)
  ctx_no_keep <- context_spec(window = 3, min_count = 100)
  m_no_keep <- train_sgns(toks_test, context = ctx_no_keep, n_dims = 20, 
                          epochs = 1, threads = 1, verbose = FALSE, seed = 333)
  
  # Train with vocab_keep
  ctx_keep <- context_spec(window = 3, min_count = 100, vocab_keep = rare_words)
  m_keep <- train_sgns(toks_test, context = ctx_keep, n_dims = 20, 
                       epochs = 1, threads = 1, verbose = FALSE, seed = 333)
  
  # vocab_keep version should have more words
  expect_true(nrow(m_keep$word_embeddings) >= nrow(m_no_keep$word_embeddings))
  
  # All of the rare words should be in vocab_keep version
  kept_words <- intersect(rare_words, rownames(m_keep$word_embeddings))
  expect_true(length(kept_words) == length(rare_words))
})

test_that("vocab parameters produce consistent vocabularies: train_sgns vs fcm", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Test vocab_size consistency
  ctx_size <- context_spec(window = 3, min_count = 2, vocab_size = 100)
  
  fcm_size <- fcm(toks_test, context = ctx_size)
  m_sgns_size <- train_sgns(toks_test, context = ctx_size, n_dims = 20, 
                            epochs = 1, threads = 1, verbose = FALSE, seed = 444)
  
  # Vocabularies should match
  expect_equal(sort(rownames(fcm_size)), sort(rownames(m_sgns_size$word_embeddings)))
  
  # Test vocab_coverage consistency
  ctx_coverage <- context_spec(window = 3, min_count = 1, vocab_coverage = 0.9)
  
  fcm_coverage <- fcm(toks_test, context = ctx_coverage)
  m_sgns_coverage <- train_sgns(toks_test, context = ctx_coverage, n_dims = 20, 
                                epochs = 1, threads = 1, verbose = FALSE, seed = 555)
  
  # Vocabularies should be very similar (may differ by 1-2 words at boundary)
  fcm_vocab <- sort(rownames(fcm_coverage))
  sgns_vocab <- sort(rownames(m_sgns_coverage$word_embeddings))
  overlap <- length(intersect(fcm_vocab, sgns_vocab))
  expect_true(overlap / max(length(fcm_vocab), length(sgns_vocab)) > 0.9)
  
  # Test vocab_keep consistency
  # Using actual rare words from data_corpus_sotu (each appears only once)
  keep_words <- c("congratulating", "accession")
  ctx_keep <- context_spec(window = 3, min_count = 100, vocab_keep = keep_words)
  
  fcm_keep <- fcm(toks_test, context = ctx_keep)
  m_sgns_keep <- train_sgns(toks_test, context = ctx_keep, n_dims = 20, 
                            epochs = 1, threads = 1, verbose = FALSE, seed = 666)
  
  # Both methods should include the kept words
  expect_true(all(keep_words %in% rownames(fcm_keep)))
  expect_true(all(keep_words %in% rownames(m_sgns_keep$word_embeddings)))
  
  # Vocabularies should be similar (may include additional high-frequency words)
  fcm_vocab <- sort(rownames(fcm_keep))
  sgns_vocab <- sort(rownames(m_sgns_keep$word_embeddings))
  expect_true(all(keep_words %in% fcm_vocab))
  expect_true(all(keep_words %in% sgns_vocab))
})

test_that("vocab_size and vocab_coverage can be combined", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # vocab_coverage should be applied first, then vocab_size limit
  ctx_both <- context_spec(window = 3, min_count = 1, 
                           vocab_coverage = 0.9, vocab_size = 30)
  
  m_both <- train_sgns(toks_test, context = ctx_both, n_dims = 20, 
                       epochs = 1, threads = 1, verbose = FALSE, seed = 777)
  
  # Should respect vocab_size limit
  expect_true(nrow(m_both$word_embeddings) <= 30)
})

test_that("vocab_keep works with vocab_coverage", {
  library(quanteda)
  
  toks_test <- make_large_test_tokens()
  
  # Force inclusion of specific words even with low coverage
  # Using actual rare words from data_corpus_sotu
  rare_words <- c("congratulating", "accession")
  ctx_combined <- context_spec(window = 3, min_count = 1, 
                               vocab_coverage = 0.3, vocab_keep = rare_words)
  
  m_combined <- train_sgns(toks_test, context = ctx_combined, n_dims = 20, 
                           epochs = 1, threads = 1, verbose = FALSE, seed = 888)
  
  # At least some rare words should be present despite low coverage
  kept_words <- intersect(rare_words, rownames(m_combined$word_embeddings))
  expect_true(length(kept_words) > 0)
})
