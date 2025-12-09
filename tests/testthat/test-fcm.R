test_that("fcm works with basic input", {
  txt <- "a b c d e"
  toks <- quanteda::tokens(txt)
  
  # Window = 1, symmetric, no decay, full matrix
  mat <- fcm(toks, window = 1, weights = "none", tri = FALSE)
  
  expect_true(sum(mat) > 0)
  # a-b co-occur
  expect_equal(as.numeric(mat["a", "b"]), 1)
  expect_equal(as.numeric(mat["b", "a"]), 1)
  # a-c do not (dist 2)
  expect_equal(as.numeric(mat["a", "c"]), 0)
})

test_that("fcm works with decay", {
  txt <- "a b c"
  toks <- quanteda::tokens(txt)
  
  # Window = 2, linear decay
  # dist(a,b) = 1. Weight = (2 - 1 + 1)/2 = 1
  # dist(a,c) = 2. Weight = (2 - 2 + 1)/2 = 0.5
  mat <- fcm(toks, window = 2, weights = "linear", tri = TRUE)
  
  expect_equal(as.numeric(mat["a", "b"]), 1)
  expect_equal(as.numeric(mat["a", "c"]), 0.5)
})

test_that("fcm works with include_target", {
  txt <- "a b"
  toks <- quanteda::tokens(txt)
  
  # Window = 1, include target
  # a-a should be 1 (if linear decay logic holds for dist 0: (1-0+1)/1 = 2? No, code says (W+1)/W for linear at 0)
  # Code: if linear, w = (window_size + 1.0) / window_size;
  # Here window=1. w = 2.
  
  mat <- fcm(toks, window = 1, weights = "linear", include_target = TRUE)
  
  expect_true(as.numeric(mat["a", "a"]) > 0)
})


test_that("fcm matches quanteda::fcm for boolean weights", {
  skip_if_not_installed("quanteda")
  
  txt <- c("a b c d e", "a b c")
  toks <- quanteda::tokens(txt)
  
  # 1. Symmetric, window=1, tri=TRUE
  # Note: quanteda::fcm default is tri=TRUE
  my_fcm <- fcm(toks, window = 1, weights = "none", tri = TRUE)
  q_fcm <- quanteda::fcm(toks, context = "window", window = 1, count = "frequency", tri = TRUE)
  
  # Compare objects directly
  # Ignore call/system info if necessary, but expect_equal handles some differences.
  # However, quanteda objects might have different creation timestamps in meta.
  # We can strip meta$system for comparison if needed.
  
  # For now, let's try direct comparison.
  # If it fails on timestamps/versions, we'll adjust.
  
  # Remove system info for comparison as it contains creation time
  my_fcm@meta$system <- NULL
  q_fcm@meta$system <- NULL
  
  expect_equal(as.matrix(my_fcm), as.matrix(q_fcm))
  
  # 2. Symmetric, window=2, tri=FALSE
  my_fcm <- fcm(toks, window = 2, weights = "none", tri = FALSE)
  q_fcm <- quanteda::fcm(toks, context = "window", window = 2, count = "frequency", tri = FALSE)
  
  expect_equal(as.matrix(my_fcm), as.matrix(q_fcm))
})

test_that("fcm matches quanteda::fcm for ordered/forward", {
  skip_if_not_installed("quanteda")
  
  txt <- "a b c d e"
  toks <- quanteda::tokens(txt)
  
  # Forward (ordered=TRUE in quanteda)
  my_fcm <- fcm(toks, window = 2, weights = "none", direction = "forward")
  q_fcm <- quanteda::fcm(toks, context = "window", window = 2, ordered = TRUE)

  expect_equal(as.matrix(my_fcm), as.matrix(q_fcm))
})

test_that("fcm matches quanteda::fcm for custom weights", {
  skip_if_not_installed("quanteda")
  
  txt <- "a b c d e"
  toks <- quanteda::tokens(txt)
  
  # Weights for window=2: dist1=1, dist2=0.5
  w <- c(1, 0.5)
  
  my_fcm <- fcm(toks, window = 2, weights = w, tri = TRUE)
  q_fcm <- quanteda::fcm(toks, context = "window", count = "weighted", window = 2, weights = w, tri = TRUE)
  
  expect_equal(as.matrix(my_fcm), as.matrix(q_fcm))
})
