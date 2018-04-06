from keras.preprocessing.text import Tokenizer
import numpy as np
import itertools as it
import argparse
import os
import json
from common.datastore import DataStore
from pandas import DataFrame
from nltk.corpus import wordnet as wn
from scipy.special import expit


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta',
                        help="Threshold for word components.",
                        required=False,
                        type=float,
                        default=0.3)
    parser.add_argument('--input-uri',
                        help="The uri to the input data.",
                        required=True)
    parser.add_argument('--num-articles',
                        help="Max number of articles to retrieve. Default is 100.",
                        type=int,
                        required=False,
                        default=100)
    parser.add_argument('--output-file',
                        help='The file in which to print the results.',
                        required=True)
    return parser.parse_args()


def load_publications(directory, ext='json'):
    assert os.path.exists(directory), '{} does not exist!'.format(directory)
    for root, dirs, files in os.walk(directory):
        for file in [f for f in files if f.endswith(ext)]:
            file_path = os.path.join(root, file)
            print(file_path)
            assert os.path.exists(file_path)
            with open(file_path) as f:
                data = json.loads(f.read())

                for item in data:
                    result = {'abstract': item['abstract'],
                              'text': item['text']}
                    yield result


def load_articles(mongodb_host, num_articles):
    ds = DataStore(host=mongodb_host)
    return ds.load_articles(num_articles)


def run(args):
    if args.input_uri.startswith('mongodb://'):
        docs = [{'doi': a.doi,
                 'abstract': a.abstract}
                for a in load_articles(args.input_uri, args.num_articles)]
    else:
        docs = [{'doi': item['doi'],
                 'abstract':item['abstract']}
                for item in load_publications(args.input_uri)]

    # create the tokehnizer
    t = Tokenizer()

    texts = [d['abstract'] for d in docs]
    # fit the tokenizer on the documents
    t.fit_on_texts(texts=texts)

    # summarize what was learned
    # print(t.word_counts)
    # print(t.document_count)
    # print(t.word_index)
    # print(t.word_docs)

    # integer encode documents
    encoded_docs = t.texts_to_matrix(texts, mode='tfidf')
    print(encoded_docs)
    data = {}
    for i, j in it.combinations(range(len(docs)), 2):
        a = encoded_docs[i]
        b = encoded_docs[j]
        a[a < args.delta] = 0
        b[b < args.delta] = 0

        enhance_with_synonyms(a, b, t.word_index)
        prod = np.multiply(a, b)
        score = np.count_nonzero(prod)
        data['{i:d}-{j:d}'.format(i=i, j=j)] = {'score': score,
                                                'i_doi': docs[i]['doi'],
                                                'j_doi': docs[j]['doi']}

    df = DataFrame.from_dict(data, orient='index')
    df.index.name = 'i-j'
    df.sort_values(by=['score'], ascending=False, inplace=True)

    # Since expit(x) is very close to 1.0 when x >= 6
    # subtract 6 from np.log(score) to have a better view of the score.
    df['score'] = expit(np.log(df.score) - 6)
    df.to_csv(args.output_file)


def enhance_with_synonyms(a, b, word_index):
    enhancer = SynonymsEnhancer(word_index)
    enhancer.enhance(a, b)


class SynonymsEnhancer:
    def __init__(self, word_index):
        self.word_index = {word: index for word, index in word_index.items()}
        self.reverse_index = {index: word for word, index in word_index.items()}

    def enhance(self, a, b):
        assert(len(a) == len(b))
        for index in range(len(a)):
            if a[index] == 0 and b[index] == 0:
                continue
            if a[index] != 0 and b[index] != 0:
                continue
            word = self.reverse_index[index]
            synonyms = self.get_synonyms(word)
            indices = [self.word_index[s] for s in synonyms if s in self.word_index]
            if not indices:
                continue
            if a[index] == 0:
                a[index] = self.compute_score(b, synonyms, indices)
            if b[index] == 0:
                b[index] = self.compute_score(a, synonyms, indices)

    def compute_score(self, document, synset, synset_indices):
        score = np.divide(
            np.sum([document[j] for j in synset_indices]),
            len(synset))
        return score

    def get_synonyms(self, word):
        ss = wn.synsets(word)
        chain = it.chain.from_iterable([w.lemma_names() for w in ss])
        s = set(chain)
        s.discard(word)
        return s


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
