import os
from argparse import ArgumentParser
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import HdpModel


def get_file_names(input_path):
    for root, dirs, files in os.walk(input_path):
        for f in files:
            yield os.path.join(root, f)


def run(args):
    vectorizer = CountVectorizer(input='filename')
    X = vectorizer.fit_transform(get_file_names(args.input_path))
    corpus = [[(idx, count) for idx, count in enumerate(doc) if count > 0]
              for doc in X.toarray()]
    dictionary = {idx: word
                  for idx, word in enumerate(vectorizer.get_feature_names())}
    hdp = HdpModel(corpus, dictionary)
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
