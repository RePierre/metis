from keras.preprocessing.text import Tokenizer
import numpy as np
import itertools as it
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta',
                        help="Threshold for word components.",
                        required=False,
                        type=float,
                        default=0.3)
    return parser.parse_args()


def run(args):
    # define 5 documents
    docs = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!']

    # create the tokenizer
    t = Tokenizer()

    # fit the tokenizer on the documents
    t.fit_on_texts(texts=docs)

    # summarize what was learned
    print(t.word_counts)
    print(t.document_count)
    print(t.word_index)
    print(t.word_docs)

    # integer encode documents
    encoded_docs = t.texts_to_matrix(docs, mode='tfidf')
    print(encoded_docs)

    for i, j in it.combinations(range(len(docs)), 2):
        a = encoded_docs[i]
        b = encoded_docs[j]
        a[a < args.delta] = 0
        b[b < args.delta] = 0

        prod = np.multiply(a, b)
        score = np.count_nonzero(prod)
        print('Similarity score between text {} and text {} is: {}'.format(i + 1, j + 1, score))


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
