
library(quanteda)
library(wordembeddings)
library(microbenchmark)

# Use real data if possible, or Zipfian synthetic data
if (FALSE) {
    # This might not be available.
    # Let's use quanteda's data_corpus_inaugural
    data("data_corpus_inaugural", package = "quanteda")
    toks <- tokens(data_corpus_inaugural, remove_punct = TRUE)
    toks <- tokens_select(toks, pattern = stopwords("en"), selection = "remove")
} else {
    # Generate Zipfian data
    set.seed(123)
    vocab_size <- 10000
    n_docs <- 1000
    doc_len <- 1000
    
    # Zipfian probabilities: 1/rank
    probs <- 1 / (1:vocab_size)
    probs <- probs / sum(probs)
    vocab <- paste0("word", 1:vocab_size)
    
    docs <- lapply(1:n_docs, function(i) {
      sample(vocab, size = doc_len, replace = TRUE, prob = probs)
    })
    toks <- as.tokens(docs)
}

cat("Number of tokens:", sum(ntoken(toks)), "\n")
cat("Benchmarking fcm()...\n")

# Run benchmark
mb <- microbenchmark(
  fcm_linear = fcm(toks, window = 5, weights = "linear", verbose = FALSE),
  fcm_none = fcm(toks, window = 5, weights = "none", verbose = FALSE),
  times = 5
)

print(mb)
