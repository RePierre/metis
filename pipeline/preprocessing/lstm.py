from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import spacy
from argparse import ArgumentParser

INPUT_SIZE = 384

nlp = spacy.load('en')


def read_text(file_path):
    result = []
    with open(file_path, 'rt', encoding='utf-8')as f:
        for line in f.readlines():
            parts = line.split('\t')
            result.append(parts[0].strip())
            result.append(parts[1].strip())
    return result


def vectorize(sentence):
    result = [t.vector for t in nlp(sentence)]
    result = np.asarray(result)
    # reshape word vectors to [1 sample, time steps, num features]
    result = np.reshape(result, (1, len(result), INPUT_SIZE))
    return result


def build_datasets(text):
    X = []
    for sentence in text:
        X.append([t.vector for t in nlp(sentence)])
    Y = [np.mean(s, axis=0) for s in X]
    X = pad_sequences(X, maxlen=INPUT_SIZE, dtype='float32')
    X = np.reshape(X, (X.shape[0], INPUT_SIZE, INPUT_SIZE))
    Y = np.asarray(Y)
    return X, Y


def run(args):
    model = Sequential()
    model.add(LSTM(INPUT_SIZE,
                   batch_input_shape=(args.batch_size, INPUT_SIZE, INPUT_SIZE),
                   stateful=args.stateful))
    model.add(Dense(INPUT_SIZE, activation='linear'))

    optimizer = RMSprop(lr=args.learning_rate)
    model.compile(loss=args.loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    X, Y = build_datasets(read_text(args.input_file))
    model.fit(X, Y, epochs=args.epochs, batch_size=args.batch_size)
    scores = model.evaluate(X, Y)
    print('Model accuracy: {:f}'.format(scores[1] * 100))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-file',
                        help='Path to the input file.',
                        required=True)
    parser.add_argument('--epochs',
                        help='Number of epochs to use for training.',
                        required=False,
                        default=500,
                        type=int)
    parser.add_argument('--batch-size',
                        help='Number of samples in a batch.',
                        required=False,
                        default=15,
                        type=int)
    parser.add_argument('--stateful',
                        help='When set will run LSTM cell in stateful mode.',
                        required=False,
                        action='store_true')
    parser.add_argument('--learning-rate',
                        help='The learning rate.',
                        required=False,
                        default=0.02,
                        type=float)
    parser.add_argument('--loss',
                        help='Loss function.',
                        required=False,
                        default='cosine_proximity')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
