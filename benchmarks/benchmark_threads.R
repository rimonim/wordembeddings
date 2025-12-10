
library(quanteda)
library(wordembeddings)
library(microbenchmark)

# Generate synthetic data
set.seed(123)
vocab_size <- 10000
n_docs <- 1000
doc_len <- 1000
vocab <- paste0("word", 1:vocab_size)
docs <- lapply(1:n_docs, function(i) {
  sample(vocab, size = doc_len, replace = TRUE)
})
toks <- as.tokens(docs)

cat("Benchmarking fcm() with different thread counts...\n")

# Run benchmark
mb <- microbenchmark(
  threads_1 = fcm(toks, window = 5, weights = "linear", threads = 1, verbose = FALSE),
  threads_2 = fcm(toks, window = 5, weights = "linear", threads = 2, verbose = FALSE),
  threads_auto = fcm(toks, window = 5, weights = "linear", threads = NULL, verbose = FALSE),
  times = 3
)

print(mb)
