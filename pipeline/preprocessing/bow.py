from keras.preprocessing.text import Tokenizer
import numpy as np

def run():
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

    for i in range(len(docs)):
        for j in range(len(docs)):
            if i != j:
                prod = np.multiply(encoded_docs[i], encoded_docs[j])
                print('Similarity score between text {} and text {} is: {}'.format(i+1, j+1, np.sum(prod)))


if __name__ == "__main__":
    run()
