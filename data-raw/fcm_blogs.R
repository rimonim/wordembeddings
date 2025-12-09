library(quanteda)
library(tidyverse)

blogs <- read_csv("/Volumes/Crucial X9/data/blogtext.csv")
blogs_corp <- blogs |> corpus()
blogs_tokens <- blogs_corp |>
	tokens(remove_punct = TRUE, remove_symbols = TRUE) |>
	tokens_tolower()

# find 100 authors with most text
keep_ids <- blogs |>
	mutate(n_token = ntoken(blogs_tokens)) |>
	group_by(id) |>
	mutate(n_token = sum(n_token)) |>
	ungroup() |>
	arrange(desc(n_token), id) |>
	pull(id) |>
	unique() |>
	head(50)

# 10,000 most frequent words overall
keep_tokens <- blogs_tokens |>
	dfm() |>
	quanteda.textstats::textstat_frequency(n = 5000) |>
	pull(feature)

fcm_blogs <- lapply(keep_ids, function(id_curr) {
	blogs_tokens |>
		tokens_subset(id == id_curr) |>
		tokens_keep(keep_tokens, valuetype = "fixed", padding = TRUE) |>
		fcm(context = "window", count = "frequency", window = 5)
})

fcm_blogs <- lapply(fcm_blogs, fcm_match, features = keep_tokens)
fcm_blogs <- do.call(abind, c(lapply(fcm_blogs, as, Class = "SVT_SparseArray"), along = 3))
dimnames(fcm_blogs)[[3]] <- keep_ids
type(fcm_blogs) <- "integer"

usethis::use_data(fcm_blogs, overwrite = TRUE)
