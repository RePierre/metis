import spacy
import textacy
from textacy import similarity as sim
import itertools as it
from argparse import ArgumentParser
from common.datastore import DataStore
from pandas import DataFrame


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--mongodb-host',
                        help='The connection string to mongodb.',
                        required=True)
    parser.add_argument('--num-articles',
                        help='The number of articles for which to create a baseline.',
                        required=False,
                        type=int,
                        default=10)
    parser.add_argument('--article-part',
                        help='The part of article to use for baseline.',
                        required=True,
                        choices=['abstract', 'text'])
    parser.add_argument('--output-file',
                        help='The file in which to save similarity scores.',
                        required=True)
    args = parser.parse_args()
    return args


def load_publications(mongodb_host, num_articles, article_part):
    ds = DataStore(host=mongodb_host)
    publications = list()
    for article in ds.load_articles(num_articles):
        text = article.abstract if article_part == 'abstract' else article.text
        publications.append({'doi': article.doi,
                             'text': textacy.preprocess_text(text, lowercase=True, no_punct=True)})
    return publications


text_similarity_metrics = [
    sim.hamming,
    sim.jaccard,
    sim.jaro_winkler,
    sim.levenshtein,
    sim.token_sort_ratio,
]

vector_similarity_metrics = [
    sim.word2vec,
    sim.word_movers
]


def run(args):
    publications = load_publications(args.mongodb_host,
                                     args.num_articles,
                                     args.article_part)
    scores = {}
    nlp = spacy.load('en')

    for i, j in it.combinations(range(len(publications)), 2):
        index_key = '{:d}-{:d}'.format(i, j)
        print("Computing similarity for pair {}.".format(index_key))
        a = publications[i]
        b = publications[j]
        scores[index_key] = {'i_doi': a['doi'],
                             'j_doi': b['doi']}
        for metric in text_similarity_metrics:
            scores[index_key][metric.__name__] = metric(a['text'], b['text'])

        embeddings_a = nlp(a['text'])
        embeddings_b = nlp(b['text'])
        for metric in vector_similarity_metrics:
            scores[index_key][metric.__name__] = metric(embeddings_a, embeddings_b)

    df = DataFrame.from_dict(scores, orient='index')
    df.index.name = 'i-j'
    df.to_csv(args.output_file)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
