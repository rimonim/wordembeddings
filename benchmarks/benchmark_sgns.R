library(quanteda)
library(embedplyr)
library(word2vec) # from https://github.com/bnosac/word2vec
library(wordembeddings)
library(microbenchmark)

# Load sample data to FCM
corp <- readLines('/Volumes/Crucial X9/data/COCA/text_fic_jjw/text_fic_2019.txt')
corp <- trimws(corp)
corp <- corp[corp != ""]
corp <- unlist(strsplit(corp, "@ @ @ @ @ @ @ @ @ @")) # split corp at removed sections (marked by "@ @ @ @ @ @ @ @ @ @")
corp <- sub("^@@\\S+\\s*", "", corp) # remove doc ids at beginning of each line (e.g. "@@221120")
toks <- quanteda::tokens(corp, remove_punct = TRUE, remove_symbols = TRUE, split_hyphens = TRUE)
toks <- quanteda::tokens_tolower(toks)

# Define context spec (reusable across methods!)
ctx <- context_spec(window = 5, weights = "none", vocab_size = 10000)  # Match word2vec's uniform weighting

# Create FCM (optional - for FCM-based training)
fcm <- wordembeddings::fcm(toks, context = ctx)
sum(fcm) # 30,614,552 training examples

# Train SGNS model (word2vec)
word2vec_mod <- word2vec(x = corp, type = "skip-gram", dim = 100, window = 5, 
                         negative = 5L, iter = 5, min_count = 5, threads = 10)
word2vec_mod <- as.embeddings(as.matrix(word2vec_mod))
cat("\nword2vec results:\n")
find_nearest(word2vec_mod, "red") |> rownames() |> dput() 

# Train SGNS model (wordembeddings - streaming from tokens)
wordembeddings_mod <- train_sgns(toks, context = ctx, 
                                 n_dims = 100, neg = 5, epochs = 5, 
                                 threads = 10, verbose = TRUE)
wordembeddings_mod <- as.embeddings(wordembeddings_mod$word_embeddings)
cat("\nwordembeddings (streaming) results:\n")
find_nearest(wordembeddings_mod, "red") |> rownames() |> dput()

# Train SGNS model (wordembeddings - streaming with weighted context)
wordembeddings_mod_weighted <- train_sgns(toks, context = context_spec(window = 5, weights = "linear", vocab_size = 10000, clean_distance = TRUE), 
                                          n_dims = 100, neg = 5, epochs = 5, 
                                          threads = 10, verbose = TRUE)
wordembeddings_mod_weighted <- as.embeddings(wordembeddings_mod_weighted$word_embeddings)
cat("\nwordembeddings (streaming weighted) results:\n")
find_nearest(wordembeddings_mod_weighted, "red") |> rownames() |> dput()

# Train SGNS model (wordembeddings - from FCM)
wordembeddings_mod_fcm <- train_sgns(fcm, n_dims = 100, neg = 5, epochs = 5, 
                                     threads = 10, verbose = TRUE)
wordembeddings_mod_fcm <- as.embeddings(wordembeddings_mod_fcm$word_embeddings)
cat("\nwordembeddings (FCM) results:\n")
find_nearest(wordembeddings_mod_fcm, "red") |> rownames() |> dput()

# Benchmark
mb <- microbenchmark(
  # word2vec = word2vec(x = corp, type = "skip-gram", dim = 100, window = 5, 
  #                     negative = 5L, iter = 5, min_count = 5, threads = 10),
  wordembeddings_streaming = train_sgns(toks, context = context_spec(window = 5, weights = "none", vocab_size = 10000, clean_distance = TRUE), 
                                        n_dims = 100, neg = 5, epochs = 5, 
                                        threads = 10, verbose = FALSE),
  wordembeddings_streaming_weighted = train_sgns(toks, context = context_spec(window = 5, weights = "linear", vocab_size = 10000, clean_distance = TRUE), 
                                               n_dims = 100, neg = 5, epochs = 5, 
                                               threads = 10, verbose = FALSE),
  wordembeddings_fcm = train_sgns(fcm, n_dims = 100, neg = 5, epochs = 5, 
                                  threads = 10, verbose = FALSE),
  times = 3
)

print(mb)

# Check that weighted context gives similar results for FCM-based and streaming
weighted_ctx <- context_spec(window = 5, weights = "linear", vocab_size = 10000)
fcm_weighted <- wordembeddings::fcm(toks, context = weighted_ctx)
wordembeddings_mod_fcm_weighted <- train_sgns(fcm_weighted, n_dims = 25, neg = 5, epochs = 10, 
                                               threads = 10, verbose = TRUE)
wordembeddings_mod_streaming_weighted <- train_sgns(toks, context = weighted_ctx, 
                                                     n_dims = 25, neg = 5, epochs = 10, 
                                                     threads = 10, verbose = TRUE)

fcm_weighted_mat <- crossprod(as.matrix(wordembeddings_mod_fcm_weighted$word_embeddings))
streaming_weighted_mat <- crossprod(as.matrix(wordembeddings_mod_streaming_weighted$word_embeddings))
cor.test(as.vector(fcm_weighted_mat), as.vector(streaming_weighted_mat)) # 0.51 I guess this is reasonable?