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


def run_experiments(args):
    df = pandas.read_csv(args.experiments_file)
    for row in df.itertuples():
        print(row)
        lstm.run(row)


if __name__ == '__main__':
    args = parse_arguments()
    run_experiments(args)
