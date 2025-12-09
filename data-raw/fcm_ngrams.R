fcm_ngrams <- readRDS("~/Projects/cultural_depolarization/data/n_grams/fcms_eng-gb.rds")[["201"]]

usethis::use_data(fcm_ngrams, overwrite = TRUE)
