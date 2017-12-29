from keras.preprocessing.text import Tokenizer
import numpy as np
import itertools as it
import argparse
import os
import json
from common.datastore import DataStore
from pandas import DataFrame


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta',
                        help="Threshold for word components.",
                        required=False,
                        type=float,
                        default=0.3)
    parser.add_argument('--input_uri',
                        help="The uri to the input data.",
                        required=True)
    parser.add_argument('--num_articles',
                        help="Max number of articles to retrieve. Default is 100.",
                        type=int,
                        required=False,
                        default=100)
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
                    result = {'abstract': item['abstract'], 'text': item['text']}
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

        prod = np.multiply(a, b)
        score = np.count_nonzero(prod)
        data['{i:d}-{j:d}'.format(i=i, j=j)] = {'score': score,
                                                'i_doi': docs[i]['doi'],
                                                'j_doi': docs[j]['doi']}

    df = DataFrame.from_dict(data, orient='index')
    df.sort_values(by=['score'], ascending=False, inplace=True)
    print(df)


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
