from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard

import numpy as np
import spacy
from argparse import ArgumentParser
import csv

INPUT_SIZE = 384

nlp = spacy.load('en')


def read_text(file_path):
    with open(file_path, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            if len(line) >= 7:
                yield (line[5], line[6], np.float32(line[4]))


def build_datasets(input):
    T1 = []
    T2 = []
    Y = []
    for sentence1, sentence2, score in input:
        T1.append([t.vector for t in nlp(sentence1)])
        T2.append([t.vector for t in nlp(sentence2)])
        Y.append(score)

    T1 = pad_sequences(T1, maxlen=INPUT_SIZE, dtype='float32')
    T1 = np.reshape(T1, (T1.shape[0], INPUT_SIZE, INPUT_SIZE))

    T2 = pad_sequences(T2, maxlen=INPUT_SIZE, dtype='float32')
    T2 = np.reshape(T2, (T2.shape[0], INPUT_SIZE, INPUT_SIZE))
    X = [T1, T2]
    Y = np.asarray(Y)
    return X, Y


def run(args):
    # Define the input nodes
    text1 = Input(shape=(INPUT_SIZE, INPUT_SIZE), name='text1')
    text2 = Input(shape=(INPUT_SIZE, INPUT_SIZE), name='text2')

    # Create the shared LSTM node
    shared_lstm = LSTM(INPUT_SIZE,
                       batch_input_shape=(args.batch_size, INPUT_SIZE, INPUT_SIZE),
                       stateful=args.stateful)

    # Run inputs through shared layer
    encoded1 = shared_lstm(text1)
    encoded2 = shared_lstm(text2)

    # Concatenate outputs
    concatenated = keras.layers.concatenate([encoded1, encoded2])

    # Create the output layer
    # It should return a single number
    output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[text1, text2], outputs=output)
    optimizer = RMSprop(lr=args.learning_rate)
    model.compile(loss=args.loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    tensorboardDisplay = TensorBoard(log_dir=args.tensorboard_log_dir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     batch_size=args.batch_size)
    X, Y = build_datasets(read_text(args.input_file))
    model.fit(X, Y, epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[tensorboardDisplay])
    scores = model.evaluate(X, Y, batch_size=args.batch_size)
    print('Model accuracy: {:f}'.format(scores[1] * 100))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--input-file',
                        help="""Path to the input file.
                        The file is expected to be a SemEval STS file
                        with the following tab separated fields
                        [genre filename year score sentence1 sentence2]""",
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
                        default='mean_squared_error')
    parser.add_argument('--tensorboard-log-dir',
                        help='The path of the directory where to save the log files to be parsed by TensorBoard',
                        required=False,
                        default='./logs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
