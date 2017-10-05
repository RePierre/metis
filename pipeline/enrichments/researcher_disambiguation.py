import os
import argparse
import logging

LOG = logging.getLogger(__name__)


def disambiguate():
    pass


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path',  help="The URL to JSON input folder",  required=True)
    argparser.add_argument('--output_path', help="The URL to JSON output folder", required=True)
    args = argparser.parse_args()

    # check paths exist
    assert os.path.exists(args.input_path),  '{} does not exist!'.format(args.input_path)
    assert os.path.exists(args.output_path), '{} does not exist!'.format(args.output_path)

    # disambiguate
    disambiguate()

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
