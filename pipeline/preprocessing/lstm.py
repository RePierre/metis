from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Lambda
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.optimizers import Adadelta, Adam, Adamax, Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.layers import concatenate
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from scipy.special import expit
from pandas import DataFrame
from random import randint

import tensorflow as tf
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


def build_datasets(input, time_steps):
    T1 = []
    T2 = []
    Y = []
    for sentence1, sentence2, score in input:
        T1.append([t.vector for t in nlp(sentence1)])
        T2.append([t.vector for t in nlp(sentence2)])
        Y.append(score)

    T1 = pad_and_reshape(T1, time_steps)
    T2 = pad_and_reshape(T2, time_steps)

    X = [T1, T2]

    # fit the scores between 0 and 1
    Y = expit(Y)
    return X, Y


def build_input_node(name, batch_size, time_steps, num_features=INPUT_SIZE):
    return Input(batch_shape=(batch_size, time_steps, num_features), name=name)


def build_optimizer(name, lr):
    ctor = optimizers[name]
    optimizer = ctor(lr=lr)
    return optimizer


def euclidean_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_true - y_pred)))


def build_model(args):
    # Define the input nodes
    text1 = build_input_node('text1', args.batch_size, args.time_steps)
    text2 = build_input_node('text2', args.batch_size, args.time_steps)

    # Create the shared LSTM node
    shared_lstm = LSTM(INPUT_SIZE, stateful=args.stateful, name='lstm1')

    # Run inputs through shared layer
    encoded1 = shared_lstm(text1)
    encoded2 = shared_lstm(text2)

    # Concatenate outputs to form a tensor of shape (2*batch_size, INPUT_SIZE)
    concatenated = concatenate([encoded1, encoded2], axis=0, name='concatenate')

    # Input shape: (2*batch_size, INPUT_SIZE)
    # Output shape: (2*batch_size, batch_size)
    dense1 = Dense(args.batch_size,
                   input_shape=(2 * args.batch_size, INPUT_SIZE),
                   activation='linear',
                   name='dense1')(concatenated)

    # Input shape: (2*batch_size, batch_size)
    # Output shape: (2*batch_size, 1)
    dense2 = Dense(1,
                   input_shape=(2 * args.batch_size, args.batch_size),
                   activation='linear',
                   name='dense2')(dense1)

    # Input shape: (2*batch_size, 1)
    # Output shape: (1, 2*batch_size)
    transpose = Lambda(lambda x: tf.transpose(x), name='transpose')(dense2)

    # Input shape: (1, 2*batch_size)
    # Output shape: (1, 1)
    dense3 = Dense(1, activation='linear', name='dense3')(transpose)

    model = Model(inputs=[text1, text2], outputs=dense3)
    optimizer = build_optimizer(name=args.optimizer, lr=args.learning_rate)
    model.compile(loss=args.loss,
                  optimizer=optimizer,
                  metrics=[euclidean_distance])
    LOG.info(model.summary())
    return model


def evaluate(model, text, X, Y, input_shape, num_batches):
    predictions = []
    for _ in range(num_batches):
        indices = [randint(0, len(text) - 1) for _ in range(input_shape[0])]
        x1 = np.reshape([X[0][i] for i in indices], input_shape)
        x2 = np.reshape([X[1][i] for i in indices], input_shape)
        y = model.predict([x1, x2])

        for i, index in enumerate(indices):
            sentence1, sentence2, score = text[index]
            predictions.append({
                "Original assigned score": score,
                "Assigned score": expit(score),
                "Predicted score": y[i][0],
            })
    df = DataFrame.from_records(predictions)
    return df


def run(args):
    LOG.info("Building model...")
    model = build_model(args)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    logdir = path.join(args.tensorboard_log_dir, current_time)
    tensorboardDisplay = TensorBoard(log_dir=logdir,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     write_grads=True,
                                     batch_size=args.batch_size)
    early_stopping = EarlyStopping(monitor='loss', patience=6)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.0002, patience=3)
    LOG.info("Building dataset...")
    text = list(read_text(args.input_file, args.num_samples))
    X, Y = build_datasets(text, args.time_steps)
    LOG.info("Done.")
    LOG.info("Fitting the model...")
    model.fit(X, Y, epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[tensorboardDisplay, reduce_lr, early_stopping])
    LOG.info("Done.")

    LOG.info("Evaluating model on whole dataset...")
    scores = model.evaluate(X, Y, batch_size=args.batch_size)
    print('Model accuracy on whole dataset: {:f}'.format(scores[1] * 100))
    LOG.info("Done.")

    LOG.info("Predicting score on first 100 pairs from dataset.")
    df = evaluate(model, text, X, Y,
                  (args.batch_size, args.time_steps, INPUT_SIZE), 100)
    df.to_csv(args.output_file)
    LOG.info("Predictions saved to {}".format(args.output_file))
    LOG.info("Done.")


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
    parser.add_argument('--output-file',
                        help='The name of the file where to save predictions.',
                        required=False,
                        default='predictions.csv')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')
    args = parse_arguments()
    run(args)
