import os
import json

import logging
from nltk.corpus import stopwords
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import HdpModel


logger = logging.getLogger(__name__)


class Corpus:
    """A generator over the corpus files."""

    def __init__(self, corpus_dir, logger=None):
        self._corpus_dir = corpus_dir
        self._text_analyzer = Corpus._build_analyzer()
        self._logger = logger

    def __iter__(self):
        for file_name in self._get_file_names():
            self._log_info("Reading file {}.".format(file_name))
            try:
                yield self._text_analyzer(get_file_text(file_name))
            except json.decoder.JSONDecodeError:
                self._log_info("Reading failed.")
                continue
            except TypeError:
                self._log_info("Reading failed.")
                continue

    def _get_file_names(self):
        for root, dirs, files in os.walk(self._corpus_dir):
            for f in files:
                yield os.path.join(root, f)

    def _build_analyzer():
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
        return vectorizer.build_analyzer()

    def _log_info(self, message):
        if self._logger:
            self._logger.info(message)


def get_file_text(file_name):
    text = []
    with open(file_name, 'rt') as f:
        data = json.load(f)
        for section in data['body']:
            title = section['title']
            contents = section['text']
            text.append(title)
            text.append(contents)
    return '\n'.join(text)


def tokenize_text(text):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    analyzer = vectorizer.build_analyzer()
    print(analyzer(text))


def run(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s',
                        filename='hdp-analysis.log',
                        filemode='w')
    logger.info("Building dictionary from corpus...")
    documents = Corpus(args.input_path, logger)
    dct = Dictionary.from_documents(documents)
    logger.info("Finished building dictionary.")

    logger.info("Running the HDP model.")
    corpus = [dct.doc2bow(doc) for doc in Corpus(args.input_path)]
    hdp = HdpModel(corpus, dct)
    hdp.optimal_ordering()
    topics = hdp.get_topics()
    num_topics, vocab_size = topics.shape
    logger.info("Found {} topics. Vocabulary size: {}".format(num_topics, vocab_size))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-path',
                        help='The directory with parsed articles.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
