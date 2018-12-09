import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from corpus import Corpus
from argparse import ArgumentParser
from logutils import create_logger
from pandas import DataFrame
import os


def run(args, logger):
    corpus = Corpus(args.corpus_dir, logger)
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(corpus)
    doc_lengths = [(np.sum(row),) for row in doc_term_matrix]
    df = DataFrame.from_records(doc_lengths, columns=['doc_length'])

    dir_name = os.path.dirname(args.output_file)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    df.to_csv(args.output_file)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--corpus-dir',
                        help='The directory with parsed articles.',
                        required=True)
    parser.add_argument('--output-file',
                        help='The file in which to store the output.',
                        required=False,
                        default='../data/corpus-analysis/article-lengths.csv')
    parser.add_argument('--log-file',
                        help='The output file for logging.',
                        required=False,
                        default='file-length-analysis.log')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    logger = create_logger('metis.file-length-analysis', args.log_file)
    run(args, logger)
