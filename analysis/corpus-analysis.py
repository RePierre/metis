import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from corpus import Corpus
from logutils import create_logger
from argparse import ArgumentParser
from pandas import DataFrame


def get_doc_term_matrix(corpus_dir, vectorizer):
    corpus = Corpus(corpus_dir)
    doc_term_matrix = vectorizer.fit_transform(corpus)
    return doc_term_matrix.toarray()


def run_tSVD_analysis(doc_term_matrix):
    t_svd = TruncatedSVD()
    t_svd.fit(doc_term_matrix)
    results = t_svd.transform(doc_term_matrix)
    return results


def run_LDA_analysis(num_topics, doc_term_matrix):
    lda = LatentDirichletAllocation(n_components=num_topics, learning_method='batch')
    results = lda.fit_transform(doc_term_matrix)
    return results


def run(args, logger):
    stop_words = 'english'

    logger.debug('Computing BoW document-term matrix.')
    bow = get_doc_term_matrix(args.corpus_dir, CountVectorizer(stop_words=stop_words))
    logger.debug('Done.')
    logger.debug('Computing TF-IDF document-term matrix.')
    tfidf = get_doc_term_matrix(args.corpus_dir, TfidfVectorizer(stop_words=stop_words))
    logger.debug('Done.')

    logger.debug('Starting Truncated SVD analysis on Bag-Of-Words representation.')
    results = run_tSVD_analysis(bow)
    logger.debug('Truncated SVD analysis finished.')
    save_results(results, args.tsvd_bow_output, logger)

    logger.debug('Starting Truncated SVD analysis on TF-IDS representation')
    results = run_tSVD_analysis(tfidf)
    logger.debug('Truncated SVD analysis finished.')
    save_results(results, args.tfidf_tsvd_output, logger)

    logger.debug('Starting LDA analysis on BoW representation with {} topics.'
                 .format(args.num_topics))
    # Clip negative values from Bag-of-Words
    results = run_LDA_analysis(args.num_topics, bow.clip(min=0))
    logger.debug('LDA analysis on BoW representation finished.')
    save_results(results, args.bow_lda_output, logger)

    logger.debug('Starting LDA analysis on TF-IDF representation with {} topics.'
                 .format(args.num_topics))
    results = run_LDA_analysis(args.num_topics, tfidf)
    logger.debug('LDA analysis on TF-IDF representation finished.')
    save_results(results, args.tfidf_lda_output, logger)

    logger.debug("That's all folks!")


def save_results(results, output_file, logger):
    logger.debug('Saving results to output file {}'.format(output_file))
    df = DataFrame.from_records(results)
    df.to_csv(output_file)
    logger.debug('Done.')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus-dir',
                        help='The directory with parsed articles.')
    parser.add_argument('--num-topics',
                        help='Number of topics for LDA analysis.',
                        default=150,
                        type=int)
    parser.add_argument('--tsvd-bow-output',
                        help='File in which to store the results of Truncated SVD analysis on BoW.',
                        required=False,
                        default='../data/corpus-analysis/bow-tsvd-output.csv')
    parser.add_argument('--tfidf-tsvd-output',
                        help='File in which to store the results of Truncated SVD analysis on TF-IDF,',
                        required=False,
                        default='../data/corpus-analysis/tfidf-tsvd-output.csv')
    parser.add_argument('--bow-lda-output',
                        help='File in which to store the results of LDA analysis on BoW representation.',
                        required=False,
                        default='../data/corpus-analysis/bow-lda-output.csv')
    parser.add_argument('--tfidf_lda_output',
                        help='File in which to store the results of LDA analysis on TF-IDF representation.',
                        required=False,
                        default='../data/corpus-analysis/tfidf-lda-output.csv')
    parser.add_argument('--log-file',
                        help='The output file for logging.',
                        required=False,
                        default='corpus-analysis.log')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    logger = create_logger('metis.corpus-analysis', args.log_file)
    run(args, logger)
