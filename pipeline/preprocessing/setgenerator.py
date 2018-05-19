import os
import csv
import numpy as np
from argparse import ArgumentParser


def save_csv(files, file_name):
    field_names = ['file1',
                   'file2',
                   'text_score']
    with open(file_name, 'wt') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for f1, f2 in files:
            writer.writerow({'file1': f1,
                             'file2': f2,
                             'text_score': None})


def run(train_dir, test_dir):
    files = os.listdir('/home/petru/Downloads/metis-in/train')
    files = files + os.listdir('/home/petru/Downloads/metis-in/test/')

    np.random.shuffle(files)
    result_set1 = files[:3000]
    result_set2 = files[3001:6000]
    zipped = list(zip(result_set1, result_set2))

    save_csv([(f1, f2) for f1, f2 in zipped[:2000]], 'set-1.csv')
    save_csv([(f1, f2) for f1, f2 in zipped[1001:3000]], 'set-2.csv')
    save_csv([(f1, f2) for f1, f2 in zipped[:1000] + zipped[2001:]], 'set-3.csv')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--test-dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args.train_dir, args.test_dir)
