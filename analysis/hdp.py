import os
import json
from nltk.corpus import stopwords
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import HdpModel


class Corpus:
    """A generator over the corpus files."""

    def __init__(self, corpus_dir):
        self._corpus_dir = corpus_dir
        self._text_analyzer = Corpus._build_analyzer()

    def __iter__(self):
        for file_name in self._get_file_names():
            try:
                yield self._text_analyzer(get_file_text(file_name))
            except json.decoder.JSONDecodeError:
                continue

    def _get_file_names(self):
        for root, dirs, files in os.walk(self._corpus_dir):
            for f in files:
                yield os.path.join(root, f)

    def _build_analyzer():
        vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
        return vectorizer.build_analyzer()


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
    documents = Corpus(args.input_path)
    dct = Dictionary.from_documents(documents)
    corpus = [dct.doc2bow(doc) for doc in Corpus(args.input_path)]
    hdp = HdpModel(corpus, dct)
    topics = hdp.get_topics()
    num_topics, vocab_size = topics.shape
    print("Found {} topics.".format(num_topics))

    print("Printing top 20 topics")
    topic_info = hdp.print_topics(num_topics=20, num_words=20)
    for _, topic in enumerate(topic_info):
        num, words = topic


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-path',
                        help='The directory with parsed articles.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
