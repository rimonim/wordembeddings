
test_that("fcm_pmi works with dense matrix", {
  m <- matrix(c(10, 0, 5, 2), nrow = 2)
  # Expected PMI calculation
  # row sums: 10, 7
  # col sums: 15, 2
  # total: 17
  # PMI(1,1) = log2(10 * 17 / (10 * 15)) = log2(17/15)
  # PMI(2,2) = log2(2 * 17 / (7 * 2)) = log2(17/7)
  
  res <- fcm_pmi(m, positive = FALSE)
  expect_equal(res[1,1], log2(17/15))
  expect_equal(res[2,2], log2(17/7))
  expect_true(is.infinite(res[2,1])) # log(0) -> -Inf
})

test_that("fcm_pmi works with sparse matrix", {
  m <- Matrix::Matrix(c(10, 0, 5, 2), nrow = 2, sparse = TRUE)
  res <- fcm_pmi(m, positive = TRUE)
  expect_s4_class(res, "sparseMatrix")
  expect_equal(res[1,1], log2(17/15))
  expect_equal(res[1,2], 0) # positive=TRUE, so 0 instead of -Inf
})

test_that("fcm_pmi handles context_smoothing parameter", {
  m <- matrix(c(10, 5, 5, 2), nrow = 2)
  # context_smoothing = 0.75 (smoothing of column sums)
  res <- fcm_pmi(m, context_smoothing = 0.75, positive = FALSE)
  expect_true(all(is.finite(res)))
  
  # Test that context_smoothing affects results
  res_no_smooth <- fcm_pmi(m, context_smoothing = 1, positive = FALSE)
  expect_false(identical(res, res_no_smooth))
})

test_that("fcm_pmi handles target_smoothing parameter", {
  m <- matrix(c(10, 5, 5, 2), nrow = 2)
  # target_smoothing = 0.75 (smoothing of row sums)
  res <- fcm_pmi(m, target_smoothing = 0.75, positive = FALSE)
  expect_true(all(is.finite(res)))
  
  # Test that target_smoothing affects results
  res_no_smooth <- fcm_pmi(m, target_smoothing = 1, positive = FALSE)
  expect_false(identical(res, res_no_smooth))
})

test_that("fcm_pmi handles both smoothing parameters together", {
  m <- matrix(c(10, 5, 5, 2), nrow = 2)
  # Both context and target smoothing
  res <- fcm_pmi(m, context_smoothing = 0.75, target_smoothing = 0.75, positive = FALSE)
  expect_true(all(is.finite(res)))
  
  # Test combinations produce different results
  res_context_only <- fcm_pmi(m, context_smoothing = 0.75, target_smoothing = 1, positive = FALSE)
  res_target_only <- fcm_pmi(m, context_smoothing = 1, target_smoothing = 0.75, positive = FALSE)
  res_both <- fcm_pmi(m, context_smoothing = 0.75, target_smoothing = 0.75, positive = FALSE)
  
  expect_false(identical(res_context_only, res_target_only))
  expect_false(identical(res_context_only, res_both))
  expect_false(identical(res_target_only, res_both))
})



test_that("fcm_pmi handles shift parameter", {
  m <- matrix(c(10, 5, 5, 2), nrow = 2)
  res_base <- fcm_pmi(m, shift = 0, positive = FALSE)
  res_shift <- fcm_pmi(m, shift = 1, positive = FALSE)
  expect_equal(res_shift, res_base + 1)
})

test_that("fcm_smooth works with laplace smoothing", {
  m <- matrix(c(1, 0, 2, 0), nrow = 2)
  res <- fcm_smooth(m, method = "laplace", estimate_zeros = TRUE)
  expect_equal(as.vector(res), c(2, 1, 3, 1))
  
  res_sparse <- fcm_smooth(m, method = "laplace", estimate_zeros = FALSE)
  expect_equal(as.vector(res_sparse), c(2, 0, 3, 0))
})

test_that("fcm_smooth works with goodturing", {
  # Need a matrix with enough counts for Good-Turing
  set.seed(123)
  counts <- sample(1:10, 100, replace = TRUE)
  m <- matrix(counts, nrow = 10)
  
  # Just check it runs and returns matrix of same dimension
  res <- fcm_smooth(m, method = "goodturing", estimate_zeros = FALSE)
  expect_equal(dim(res), dim(m))
  expect_true(all(res >= 0))
})

test_that("fcm_positive removes negative values", {
  m <- matrix(c(1, -1, 2, -2), nrow = 2)
  res <- fcm_positive(m)
  expect_equal(as.vector(res), c(1, 0, 2, 0))
  
  sm <- Matrix::Matrix(c(1, -1, 2, -2), nrow = 2, sparse = TRUE)
  res_sm <- fcm_positive(sm)
  expect_equal(as.vector(res_sm), c(1, 0, 2, 0))
  expect_s4_class(res_sm, "sparseMatrix")
})

test_that("fcm_log works correctly", {
  m <- matrix(c(1, 2, 4, 8), nrow = 2)
  res <- fcm_log(m, base = 2)
  expect_equal(as.vector(res), c(0, 1, 2, 3))
  
  # Test positive=TRUE behavior with small values
  m2 <- matrix(c(0.5, 2), nrow = 1)
  res2 <- fcm_log(m2, positive = TRUE, base = 2)
  # 0.5 should be treated as 1 (log(1)=0) if positive=TRUE logic holds for values < 1
  expect_equal(res2[1,1], 0) 
})

test_that("fcm_log handles sparse matrices", {
  sm <- Matrix::Matrix(c(0, 4, 8, 0), nrow = 2, sparse = TRUE)
  res <- fcm_log(sm, base = 2, positive = TRUE)
  expect_s4_class(res, "sparseMatrix")
  expect_equal(res[1,2], 3)
  expect_equal(res[2,1], 2)
  expect_equal(res[1,1], 0)
})

test_that("fcm_pmi handles 3D arrays", {
  a <- array(c(10, 0, 5, 2, 10, 0, 5, 2), dim = c(2, 2, 2))
  res <- fcm_pmi(a, positive = FALSE)
  expect_equal(dim(res), c(2, 2, 2))
  expect_equal(res[,,1], res[,,2])
})
