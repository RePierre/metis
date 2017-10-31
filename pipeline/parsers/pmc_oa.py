import os
import json
import argparse
import logging
from fileparser import FileParser

LOG = logging.getLogger(__name__)


def parse_files(xml_path):
    parser = FileParser(LOG)
    return parser.parse_files(xml_path)


def store_output(pubs, output_path):
    for idx, pub in enumerate(pubs):
        with open(os.path.join(output_path, str(idx) + '.json'), 'w') as f:
            f.write(json.dumps(pub, indent=True, sort_keys=True, ensure_ascii=False))


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', help="The URL to XML input folder", required=True)
    argparser.add_argument('--output_path', help="The URL to JSON output folder", required=True)
    args = argparser.parse_args()

    # check paths exist
    assert os.path.exists(args.input_path), '{} does not exist!'.format(args.input_path)
    assert os.path.exists(args.output_path), '{} does not exist!'.format(args.output_path)

    # do the actual parsing
    pubs = parse_files(args.input_path)
    store_output(pubs, args.output_path)

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
