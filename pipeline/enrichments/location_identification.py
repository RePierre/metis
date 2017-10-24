import os
import csv
import json
import argparse
import logging
from difflib import SequenceMatcher

LOG = logging.getLogger(__name__)


def csv_dict_list(grid_data_path):
    """
    Read GRID data from .csv to a dict of org_info items.
    :param grid_data_path: Path to where GRID data is stored.
    :return: dictionary of grid info items.
    """
    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
    reader = csv.DictReader(open(grid_data_path, 'r'))
    grid_info_all = dict()
    for line in reader:
        # check there are no key duplicates
        assert line['ID'] not in grid_info_all.keys(), 'key {} already stored'.format(line['ID'])
        # dict keys are org names
        grid_info_all[line['ID']] = line
    return grid_info_all


def string_similarity_score(a, b):
    """
    Computes similarity score between string "a" and "b".
    :param a: arg "a" is the first of two sequences to be compared.
    :param b: arg "b" is the second of two sequences to be compared.
    :return: Similarity score. A value above 0.6 indicates a good similarity between the two.
    """
    return SequenceMatcher(None, a, b).ratio()

def country_name_expand(country_name):
    if country_name == 'USA':
        return 'United States'
    elif country_name == 'UK':
        return 'United Kingdom'
    else:
        return country_name

def match_countries(c1, c2):
    c1 = country_name_expand(c1)
    c2 = country_name_expand(c2)
    # check strings are matching
    if c1 and c2 and c1 == c2:
        return True
    else:
        return False


def get_grid_info(grid_org_data, raw_string, country):
    """
    Retrieve GRID org info if a match was found.
    :param grid_org_data: dictionary of grid info items.
    :param raw_string: organization raw string (includes researcher name & affiliation)
    :param country: organization country
    :return: GRID info
    """
    res = dict()
    # check raw string is non-empty
    if raw_string:
        for k, v in grid_org_data.items():
            # As a rule of thumb, a .ratio() value over 0.6 means the sequences are close matches
            if v['Name'] in raw_string and string_similarity_score(v['Name'], raw_string) > 0.5 and match_countries(country, v['Country']):
                res = v
                break
    return res


def get_grid_enhanced_pubs(grid_org_data, input_path):
    """
    Identify GRID org locations based on string matching
    :param grid_org_data: list of GRID org information.
    :param input_path: path from where to read raw publications.
    :return:
    """
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                LOG.info('Parsing {}'.format(file_path))
                with open(file_path, 'r') as json_file:
                    pub_dict = json.loads(json_file.read())
                    authors = pub_dict['authors']
                    for idx, a in enumerate(authors):
                        pub_dict['authors'][idx]['grid_info'] = get_grid_info(grid_org_data, a.get('raw'), a.get('country'))
                    # method is a generator to optimize for memory consumption.
                    yield pub_dict


def store_pubs(pubs, output_path):
    """
    Stores publications in json files.
    :param pubs: list of publications.
    :param output_path: path where to store .json files
    :return:
    """
    for pub in pubs:
        assert pub.get('pmcid'), 'Publication is missing its pmcid'
        file_path = os.path.join(output_path, '{}.json'.format(pub['pmcid']))
        with open(file_path, 'w') as f:
            LOG.info('Storing output to {}'.format(file_path))
            # ensure_ascii = False to format UTF-8 chars correctly.
            f.write(json.dumps(pub, indent=True, sort_keys=True, ensure_ascii=False))


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path',     help="The path to JSON input folder",  required=True)
    argparser.add_argument('--output_path',    help="The path to JSON output folder", required=True)
    argparser.add_argument('--grid_data_path', help="The path to GRID data folder",   required=True)
    args = argparser.parse_args()

    # check paths exist
    assert os.path.exists(args.input_path),     '{} does not exist!'.format(args.input_path)
    assert os.path.exists(args.output_path),    '{} does not exist!'.format(args.output_path)
    assert os.path.exists(args.grid_data_path), '{} does not exist!'.format(args.grid_data_path)

    # read GRID location data into a list of dictionaries
    grid_org_data = csv_dict_list(args.grid_data_path)

    # identify location based on GRID
    pubs = get_grid_enhanced_pubs(grid_org_data, args.input_path)

    # store enriched publications as .json
    store_pubs(pubs, args.output_path)

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
