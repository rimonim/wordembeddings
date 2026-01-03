test_that("train_glove returns correct structure", {
  skip_if_not_installed("quanteda")
  
  # Create a simple FCM for testing
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3, 4, 4, 5),
    j = c(2, 3, 1, 3, 1, 2, 5, 3, 4),
    x = c(5, 3, 5, 4, 3, 4, 2, 1, 2),
    dims = c(5, 5),
    dimnames = list(
      c("the", "cat", "sat", "on", "mat"),
      c("the", "cat", "sat", "on", "mat")
    )
  )
  # Make symmetric
  fcm <- fcm + Matrix::t(fcm)
  
  # Test with output = "all" to get both embeddings
  result <- train_glove(
    fcm,
    n_dims = 10,
    epochs = 5,
    output = "all",
    verbose = FALSE,
    threads = 1
  )
  
  expect_type(result, "list")
  expect_true("word_embeddings" %in% names(result))
  expect_true("context_embeddings" %in% names(result))
  expect_true("bias_i" %in% names(result))
  expect_true("bias_j" %in% names(result))
  expect_true("cost_history" %in% names(result))
  expect_equal(nrow(result$word_embeddings), 5)
  expect_equal(ncol(result$word_embeddings), 10)
  expect_equal(nrow(result$context_embeddings), 5)
  expect_equal(ncol(result$context_embeddings), 10)
  expect_equal(rownames(result$word_embeddings), c("the", "cat", "sat", "on", "mat"))
  expect_equal(rownames(result$context_embeddings), c("the", "cat", "sat", "on", "mat"))
})

test_that("train_glove output parameter works", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(5, 3, 5, 4, 3, 4),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  # Test word_embeddings only
  result_word <- train_glove(fcm, n_dims = 5, epochs = 3, 
                              output = "word_embeddings", verbose = FALSE, threads = 1)
  expect_null(result_word$context_embeddings)
  expect_false(is.null(result_word$word_embeddings))
  

  # Test context_embeddings only
  result_context <- train_glove(fcm, n_dims = 5, epochs = 3, 
                                 output = "context_embeddings", verbose = FALSE, threads = 1)
  expect_null(result_context$word_embeddings)
  expect_false(is.null(result_context$context_embeddings))
  
  # Test all
  result_all <- train_glove(fcm, n_dims = 5, epochs = 3, 
                             output = "all", verbose = FALSE, threads = 1)
  expect_false(is.null(result_all$word_embeddings))
  expect_false(is.null(result_all$context_embeddings))
})

test_that("train_glove weighting functions work", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(50, 30, 50, 40, 30, 40),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  # Test different weighting functions don't error
  for (wf in c("glove", "power", "log", "uniform")) {
    result <- train_glove(fcm, n_dims = 5, epochs = 3, 
                           weight_fn = wf, verbose = FALSE, threads = 1)
    expect_type(result, "list")
    expect_false(any(is.na(result$word_embeddings)))
  }
})

test_that("train_glove fix_bias parameter works", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(5, 3, 5, 4, 3, 4),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  # Should run without error with fix_bias = TRUE
  result_fixed <- train_glove(fcm, n_dims = 5, epochs = 3, 
                               fix_bias = TRUE, verbose = FALSE, threads = 1)
  expect_type(result_fixed, "list")
  expect_false(any(is.na(result_fixed$word_embeddings)))
  
  # Results should differ from non-fixed bias
  result_free <- train_glove(fcm, n_dims = 5, epochs = 3, 
                              fix_bias = FALSE, verbose = FALSE, threads = 1,
                              seed = 42)
  result_fixed2 <- train_glove(fcm, n_dims = 5, epochs = 3, 
                                fix_bias = TRUE, verbose = FALSE, threads = 1,
                                seed = 42)
  # The embeddings should be different when bias is fixed vs free
  # (using same seed for initialization)
  expect_false(identical(result_free$word_embeddings, result_fixed2$word_embeddings))
})

test_that("train_glove seed produces reproducible results", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(5, 3, 5, 4, 3, 4),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  result1 <- train_glove(fcm, n_dims = 5, epochs = 5, seed = 123, 
                          verbose = FALSE, threads = 1)
  result2 <- train_glove(fcm, n_dims = 5, epochs = 5, seed = 123, 
                          verbose = FALSE, threads = 1)
  
  expect_equal(result1$word_embeddings, result2$word_embeddings)
  expect_equal(result1$context_embeddings, result2$context_embeddings)
})

test_that("train_glove handles different initialization methods", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(5, 3, 5, 4, 3, 4),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  result_uniform <- train_glove(fcm, n_dims = 5, epochs = 3, 
                                 init = "uniform", verbose = FALSE, threads = 1)
  result_normal <- train_glove(fcm, n_dims = 5, epochs = 3, 
                                init = "normal", verbose = FALSE, threads = 1)
  
  expect_type(result_uniform, "list")
  expect_type(result_normal, "list")
})

test_that("train_glove works with quanteda fcm", {
  skip_if_not_installed("quanteda")
  
  # Create tokens and FCM using quanteda
  toks <- quanteda::tokens(c("the cat sat on the mat",
                              "the dog ran on the grass",
                              "the cat and dog played"))
  fcm <- quanteda::fcm(toks, context = "window", window = 2)
  
  result <- train_glove(fcm, n_dims = 10, epochs = 5, 
                         verbose = FALSE, threads = 1)
  
  expect_type(result, "list")
  expect_equal(nrow(result$word_embeddings), nrow(fcm))
  expect_equal(ncol(result$word_embeddings), 10)
  expect_equal(rownames(result$word_embeddings), rownames(fcm))
})

test_that("train_glove handles 3D arrays", {
  skip_if_not_installed("SparseArray")
  
  # Create two 2D FCMs
  fcm1 <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(5, 3, 5, 4, 3, 4),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm1 <- fcm1 + Matrix::t(fcm1)
  
  fcm2 <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(10, 6, 10, 8, 6, 8),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm2 <- fcm2 + Matrix::t(fcm2)
  
  # Stack into 3D array
  fcm_3d <- S4Arrays::abind(
    as.array(as.matrix(fcm1)),
    as.array(as.matrix(fcm2)),
    along = 3
  )
  dimnames(fcm_3d) <- list(c("a", "b", "c"), c("a", "b", "c"), c("time1", "time2"))
  
  result <- train_glove(fcm_3d, n_dims = 5, epochs = 3, 
                         verbose = FALSE, threads = 1)
  
  expect_type(result, "list")
  expect_equal(dim(result$word_embeddings), c(3, 5, 2))
  expect_equal(dimnames(result$word_embeddings)[[3]], c("time1", "time2"))
})

test_that("train_glove validates input parameters",
{
  fcm <- Matrix::sparseMatrix(
    i = c(1, 2),
    j = c(2, 1),
    x = c(5, 5),
    dims = c(2, 2),
    dimnames = list(c("a", "b"), c("a", "b"))
  )
  
  # Invalid n_dims

expect_error(train_glove(fcm, n_dims = 0))
  expect_error(train_glove(fcm, n_dims = -1))
  
  # Invalid epochs
  expect_error(train_glove(fcm, epochs = 0))
  
  # Invalid learning rate
  expect_error(train_glove(fcm, lr = 0))
  expect_error(train_glove(fcm, lr = -0.1))
  
  # Invalid weight_fn
  expect_error(train_glove(fcm, weight_fn = "invalid"))
  
  # Invalid init
  expect_error(train_glove(fcm, init = "invalid"))
  
  # Invalid output
  expect_error(train_glove(fcm, output = "invalid"))
})

test_that("train_glove cost decreases over epochs", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3, 4, 4),
    j = c(2, 3, 1, 3, 1, 2, 2, 3),
    x = c(50, 30, 50, 40, 30, 40, 20, 15),
    dims = c(4, 4),
    dimnames = list(c("a", "b", "c", "d"), c("a", "b", "c", "d"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  # Capture verbose output to check cost is decreasing
  # For now, just verify it runs without error with more epochs
  result <- train_glove(fcm, n_dims = 10, epochs = 20, 
                         verbose = FALSE, threads = 1, seed = 42)
  
  expect_type(result, "list")
  # Embeddings should be finite
  expect_true(all(is.finite(result$word_embeddings)))
  expect_true(all(is.finite(result$context_embeddings)))
})

test_that("train_glove x_max and alpha parameters affect results", {
  fcm <- Matrix::sparseMatrix(
    i = c(1, 1, 2, 2, 3, 3),
    j = c(2, 3, 1, 3, 1, 2),
    x = c(100, 30, 100, 40, 30, 40),
    dims = c(3, 3),
    dimnames = list(c("a", "b", "c"), c("a", "b", "c"))
  )
  fcm <- fcm + Matrix::t(fcm)
  
  # Different x_max values should produce different results
  result1 <- train_glove(fcm, n_dims = 5, epochs = 5, x_max = 10, 
                          seed = 42, verbose = FALSE, threads = 1)
  result2 <- train_glove(fcm, n_dims = 5, epochs = 5, x_max = 200, 
                          seed = 42, verbose = FALSE, threads = 1)
  
  # Results should differ due to different weighting
  expect_false(identical(result1$word_embeddings, result2$word_embeddings))
  
  # Different alpha values should produce different results  
  result3 <- train_glove(fcm, n_dims = 5, epochs = 5, alpha = 0.5, 
                          seed = 42, verbose = FALSE, threads = 1)
  result4 <- train_glove(fcm, n_dims = 5, epochs = 5, alpha = 1.0, 
                          seed = 42, verbose = FALSE, threads = 1)
  
  expect_false(identical(result3$word_embeddings, result4$word_embeddings))
})
