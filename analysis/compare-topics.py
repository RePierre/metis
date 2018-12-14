from argparse import ArgumentParser
import re

pattern = re.compile('Topic #(?P<num>\d+):(?P<words>.+)')


def parse_file(file_name):
    with open(file_name) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                num = int(m.group('num'))
                words = m.group('words').strip().split()
                yield (num, words)


def run(args):
    return


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--lda-log-file',
                        help='The log file of the LDA analysis.',
                        required=True)
    parser.add_argument('--nmf-log-file',
                        help='The log file of the NMF analysis.',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run(args)
