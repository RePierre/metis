from argparse import ArgumentParser
from pandas import DataFrame
import training.lstm as lstm
import pandas


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--experiments-file',
                        help='Path to the file defining experiments.',
                        default='experiments.csv',
                        required=False)
    return parser.parse_args()


def plot_predictions(predictions_file):
    def plot_to_image(dataframe, column, image):
        df = dataframe[column]
        plt = df.plot()
        plt.save(image)
    df = pandas.read_csv(predictions_file)
    plot_to_image(df, 'Original score',
                  '{}.originalscore.png'.format(predictions_file))
    plot_to_image(df, 'Predicted score',
                  '{}.predictedscore.png'.format(predictions_file))


def run_experiments(args):
    df = pandas.read_csv(args.experiments_file)
    for row in df.itertuples():
        print(row)
        lstm.run(row)


if __name__ == '__main__':
    args = parse_arguments()
    run_experiments(args)
