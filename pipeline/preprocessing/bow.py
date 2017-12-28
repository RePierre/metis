from keras.preprocessing.text import Tokenizer


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


if __name__ == "__main__":
    run()
