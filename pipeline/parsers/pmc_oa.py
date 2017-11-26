import os
import json
import argparse
import logging
from fileparser import FileParser
from datastore import DataStore

LOG = logging.getLogger(__name__)


def parse_files(xml_path, debug_mode):
    parser = FileParser(debug_mode)
    return parser.parse_files(xml_path)


def store_output_to_disk(pubs, output_path):
    for idx, pub in enumerate(pubs):
        with open(os.path.join(output_path, str(idx) + '.json'), 'w') as f:
            f.write(json.dumps(pub, indent=True, sort_keys=True, ensure_ascii=False))


def store_output_to_mongo(pubs, host):
    ds = DataStore(host=host)
    ds.store_publications(pubs)


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', help="The URL to XML input folder", required=True)
    argparser.add_argument('--output_path', help="The URL to JSON output folder", required=False)
    argparser.add_argument('--mongodb_host', help="The host URI to Mongo DB", required=False)
    argparser.add_argument('--debug', help="Run in debug mode; will delete files that are processed", required=False)
    args = argparser.parse_args()

    # check paths exist
    assert os.path.exists(args.input_path), '{} does not exist!'.format(args.input_path)
    assert args.output_path or args.mongodb_host, 'You must specify either output path or host string for Mongo DB.'
    if args.output_path:
        assert os.path.exists(args.output_path), '{} does not exist!'.format(args.output_path)

    # do the actual parsing
    pubs = parse_files(args.input_path, args.debug)
    if args.output_path:
        store_output_to_disk(pubs, args.output_path)
    else:
        store_output_to_mongo(pubs, args.mongodb_host)

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
