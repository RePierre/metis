import json
import os
import sys
import logging
from argparse import ArgumentParser

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger('metis.lda-analysis')
file_log_handler = logging.FileHandler('logfile.log')
logger.addHandler(file_log_handler)
stdout_log_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stdout_log_handler.setFormatter(formatter)

logger.setLevel('DEBUG')


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        logger.info(message)


def read_text(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            try:
                with open(file_name, 'rt') as f:
                    data = json.load(f)
                    text = []
                    for section in data['body']:
                        text.append(section['title'])
                        text.append(section['text'])
                    yield ''.join(text)
            except json.decoder.JSONDecodeError:
                continue
            except TypeError:
                continue


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-path', help='The directory with parsed articles.')
    parser.add_argument('--num-topics',
                        help='Number of topics to infer.',
                        required=False,
                        default=10,
                        type=int)
    parser.add_argument('--num-features',
                        help='Number of features.',
                        required=False,
                        default=1000,
                        type=int)
    parser.add_argument('--num-top-words',
                        help='Number of top words to print.',
                        required=False,
                        default=20,
                        type=int)
    args = parser.parse_args()
    return args


def run(path, n_components, n_features=1000, n_top_words=20):
    data_samples = list(read_text(path))
    n_samples = len(data_samples)
    logger.debug('Number of samples: {}'.format(n_samples))

    logger.debug('Started calculating term frequencies.')
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(data_samples)
    logger.debug('Finished calculating term frequencies.')

    logger.debug('Started LDA analysis.')
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()
    logger.debug('Finished LDA analysis.')

    print_top_words(lda, tf_feature_names, n_top_words)


if __name__ == '__main__':
    args = parse_arguments()
    run(args.input_path, args.num_topics, args.num_features, args.num_top_words)
