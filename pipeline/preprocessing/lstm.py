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
from pandas import DataFrame
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import spacy
from argparse import ArgumentParser
import csv
import sys
import datetime
import os.path as path
import logging
import math

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


def build_datasets(data, batch_size, time_steps):
    t1, t2, y = [], [], []
    for sentence1, sentence2, score in data:
        t1.append([t.vector for t in nlp(sentence1)])
        t2.append([t.vector for t in nlp(sentence2)])
        y.append(score)

    t1 = pad_and_reshape(t1, time_steps)
    t2 = pad_and_reshape(t2, time_steps)
    y = np.asarray(y)

    # Split dataset into train/test
    t1_train, t1_test, t2_train, t2_test, y_train, y_test = train_test_split(t1, t2, y)

    # Adjust collection sizes to have full batches of data
    x_train = [adjust_size(t1_train, batch_size),
               adjust_size(t2_train, batch_size)]
    x_test = [adjust_size(t1_test, batch_size),
              adjust_size(t2_test, batch_size)]
    y_train = adjust_size(y_train, batch_size)
    y_test = adjust_size(y_test, batch_size)

    return x_train, x_test, y_train, y_test


def adjust_size(collection, batch_size):
    num_samples = int(len(collection) / batch_size) * batch_size
    return collection[:num_samples]


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
    batch_size = input_shape[0]
    num_slices = math.ceil(len(Y) / batch_size)
    for x1, x2, y in zip(np.array_split(X[0], num_slices),
                         np.array_split(X[1], num_slices),
                         np.array_split(Y, num_slices)):
        x = [np.reshape(x1, input_shape),
             np.reshape(x1, input_shape)]
        y_ = model.predict(x)
        for i in range(len(y_)):
            predictions.append({
                "Original score": y[i],
                "Predicted score": y_[i][0]
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
    early_stopping = EarlyStopping(monitor='loss',
                                   patience=args.early_stopping_patience)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=args.reduce_lr_factor,
                                  patience=args.reduce_lr_patience)
    LOG.info("Building dataset...")
    text = list(read_text(args.input_file, args.num_samples))
    x_train, x_test, y_train, y_test = build_datasets(text, args.batch_size, args.time_steps)
    LOG.info("Done.")
    LOG.info("Fitting the model...")
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[tensorboardDisplay, reduce_lr, early_stopping])
    LOG.info("Done.")

    LOG.info("Evaluating model...")
    scores = model.evaluate(x_test, y_test, batch_size=args.batch_size)
    print('Model accuracy: {:f}'.format(scores[1] * 100))
    LOG.info("Done.")

    LOG.info("Predicting score on test split...")
    df = evaluate(model, text, x_test, y_test,
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
    parser.add_argument('--reduce-lr-factor',
                        help='The factor parameter for ReduceLROnPlateau callback.',
                        required=False,
                        default=0.0002,
                        type=float)
    parser.add_argument('--reduce-lr-patience',
                        help='The patience parameter for ReduceLROnPlateau callback.',
                        required=False,
                        default=30,
                        type=int)
    parser.add_argument('--early-stopping-patience',
                        help='The patience parameter for EarlyStopping callback.',
                        required=False,
                        default=60,
                        type=int)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')
    args = parse_arguments()
    run(args)
