from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.optimizers import Adadelta, Adam, Adamax, Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
from keras.callbacks import TensorBoard
from scipy.special import expit

import numpy as np
import spacy
from argparse import ArgumentParser
import csv
import sys
import datetime
import os.path as path
import logging

LOG = logging.getLogger(__name__)
INPUT_SIZE = 384

nlp = spacy.load('en')
optimizers = {
    'sgd': SGD,
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'adam': Adam,
    'adamax': Adamax,
    'nadam': Nadam
}


def read_text(file_path, num_samples):
    with open(file_path, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for number, line in enumerate(reader):
            if len(line) >= 7 and number < num_samples:
                yield (line[5], line[6], np.float32(line[4]))


def pad_and_reshape(sequence, time_steps):
    sequence = pad_sequences(sequence, maxlen=time_steps, dtype='float32')
    num_samples, num_features = sequence.shape[0], INPUT_SIZE
    sequence = np.reshape(sequence, (num_samples, time_steps, num_features))
    return sequence


def build_datasets(input, time_steps, output_shape):
    T1 = []
    T2 = []
    Y = []
    for sentence1, sentence2, score in input:
        T1.append([t.vector for t in nlp(sentence1)])
        T2.append([t.vector for t in nlp(sentence2)])
        Y.append(np.full(output_shape, score))

    T1 = pad_and_reshape(T1, time_steps)
    T2 = pad_and_reshape(T2, time_steps)

    X = [T1, T2]
    Y = np.asarray(Y)
    # fit the scores between 0 and 1
    Y = expit(Y)
    return X, Y


def build_input_node(name, batch_size, time_steps, num_features=INPUT_SIZE):
    return Input(batch_shape=(batch_size, time_steps, num_features), name=name)


def build_optimizer(name, lr):
    ctor = optimizers[name]
    optimizer = ctor(lr=lr)
    return optimizer


def build_model(args):
    # Define the input nodes
    text1 = build_input_node('text1', args.batch_size, args.time_steps)
    text2 = build_input_node('text2', args.batch_size, args.time_steps)

    # Create the shared LSTM node
    shared_lstm = LSTM(INPUT_SIZE, stateful=args.stateful)

    # Run inputs through shared layer
    encoded1 = shared_lstm(text1)
    encoded2 = shared_lstm(text2)

    # Concatenate outputs to form a tensor of shape (2*batch_size, INPUT_SIZE)
    concatenated = concatenate([encoded1, encoded2], axis=0)

    # Input shape: (2*batch_size, INPUT_SIZE)
    # Output shape: (2*batch_size, batch_size)
    dense1 = Dense(args.batch_size,
                   input_shape=(2 * args.batch_size, INPUT_SIZE),
                   activation='sigmoid')(concatenated)

    # Input shape: (2*batch_size, batch_size)
    # Output shape: (2*batch_size, 1)
    output_shape = (2 * args.batch_size, 1)
    output = Dense(1,
                   input_shape=(2 * args.batch_size, args.batch_size),
                   activation='sigmoid')(dense1)

    model = Model(inputs=[text1, text2], outputs=output)
    optimizer = build_optimizer(name=args.optimizer, lr=args.learning_rate)
    model.compile(loss=args.loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('''
    Model shapes:
    [concatenated]: {},
    [dense1]: {},
    [output]: {}'''.format(
        concatenated.get_shape(),
        dense1.get_shape(),
        output.get_shape()
    ))
    return model, output_shape


def run(args):
    LOG.info("Building model...")
    model, output_shape = build_model(args)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    logdir = path.join(args.tensorboard_log_dir, current_time)
    tensorboardDisplay = TensorBoard(log_dir=logdir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     batch_size=args.batch_size)
    LOG.info("Building dataset...")
    text = read_text(args.input_file, args.num_samples)
    X, Y = build_datasets(text, args.time_steps, output_shape)
    LOG.info("Done.")
    LOG.info("Fitting the model...")
    model.fit(X, Y, epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[tensorboardDisplay])
    LOG.info("Done.")
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
                        default=4,
                        type=int)
    parser.add_argument('--time-steps',
                        help='Number of time steps from each series.',
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
                        default=0.002,
                        type=float)
    parser.add_argument('--num-samples',
                        help='Maximum number of samples to read from the input file.',
                        required=False,
                        default=sys.maxsize,
                        type=int)
    parser.add_argument('--optimizer',
                        help="The optimizer to use for training.",
                        required=False,
                        default='adam')
    parser.add_argument('--loss',
                        help='Loss function.',
                        required=False,
                        default='logcosh')
    parser.add_argument('--tensorboard-log-dir',
                        help='The path of the directory where to save the log files to be parsed by TensorBoard',
                        required=False,
                        default='./logs')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')
    args = parse_arguments()
    run(args)
