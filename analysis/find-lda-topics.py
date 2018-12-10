from corpus import Corpus
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Create a custom LemmaTokenizer as specified here https://scikit-learn.org/stable/modules/feature_extraction.html


class LemmaTokenizer:
    def __init__(self):
        self._wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self._wnl.lemmatize(t) for t in word_tokenize(doc)]


def run(args, logger):
    corpus = Corpus(args.corpus_dir)
    logger.debug('Computing document-term matrix....')
    vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                 stop_words='english',
                                 lowercase=True)
    doc_term_matrix = vectorizer.fit_transform(corpus)
    logger.debug('Document-term matrix computed.')

    logger.debug('Building search parameters...')
    search_params = {
        'n_components': build_num_components(args.n_components_start,
                                             args.n_components_end,
                                             args.n_components_step),
        'learning_decay': build_learning_decay(args.learning_decay_start,
                                               args.learning_decay_end,
                                               args.learning_decay_step)
    }
    logger.debug('Finished building search parameters.')

    logger.debug('Starting grid search for number of topics...')
    model = GridSearchCV(
        LatentDirichletAllocation(batch_size=256),
        cv=None,
        n_jobs=-1,
        param_grid=search_params
    )
    logger.debug('Finished grid search.')

    logger.debug('Saving results to output file {}.'
                 .format(args.output_file))
    logger.debug('Results saved.')
    logger.debug("That's all folks!")


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
                        default='find-num-topics.log')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    logger = create_logger('metis.lda-find-num-topics', args.log_file)
    run(args, logger)
