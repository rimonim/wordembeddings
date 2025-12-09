#' Feature co-occurrence matrix from the Blog Authorship Corpus
#'
#' A 3-dimensional [SVT_SparseArray][SparseArray::SVT_SparseArray] (10,000 × 10,000 × 100)
#' representing a stack of 100 feature-co-occurence matrices. Co-occurrence data
#' were collected from the 100 authors with most total tokens in the Blog
#' Authorship Corpus (Schler et al., 2006), with an unweighted symmetrical
#' window of 5 words. The matrices are restricted to the 10,000 most common
#' tokens in the full Blog Authorship Corpus.
#'
#' @references J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006).
#'	 Effects of Age and Gender on Blogging. in Proceedings of 2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs.
#'   https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
#' @keywords data fcm_blogs
"fcm_blogs"
