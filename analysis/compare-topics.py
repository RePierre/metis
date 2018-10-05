from argparse import ArgumentParser


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
