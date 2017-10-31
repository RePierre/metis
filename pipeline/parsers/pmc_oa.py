import os
import json
import gzip
import argparse
import logging
import xmltodict as xd
from articleparser import ArticleParser
from dictquery import DictQuery

LOG = logging.getLogger(__name__)


def to_structured_data(raw_pubs, xpaths):
    pubs = list()
    for rp in raw_pubs:
        s_pub = dict()
        # query for paths
        for key, path in xpaths:
            query_result = DictQuery(rp)
            s_pub[key] = query_result.get(path)
        for v_item in zip(*[s_pub[k] for k in s_pub.keys()]):
            tmp_dict = dict()
            for idx, k in enumerate(s_pub.keys()):
                tmp_dict[k] = v_item[idx]
            pubs.append(tmp_dict)
    return pubs


def parse_files(xml_path):

    xpaths = [
        ('article_title', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/title-group/article-title'),
        ('journal_title', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/journal-title-group/journal-title'),
        ('doi', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/article-id'),
        ('publisher_name', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-name'),
        ('publisher_location', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-loc'),
        ('keywords', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/kwd-group'),
        ('abstract', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/abstract/p'),
        ('authors', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/author-notes/fn'),
        ('acknowledgement', 'OAI-PMH/ListRecords/record/metadata/article/back/ack/p'),
        ('identifier', 'OAI-PMH/ListRecords/record/header/identifier'),
        ('body', 'OAI-PMH/ListRecords/record/metadata/article/body/sec')
    ]

    raw_pubs = list()
    parser = ArticleParser()
    for root, dirs, files in os.walk(xml_path):
        for file in files:
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                LOG.info('Parsing {}'.format(file_path))
                with gzip.open(file_path, 'r') as gzip_file:
                    pub_xml = gzip_file.read()
                    pub_dict = xd.parse(pub_xml)
                    raw_pubs.append(pub_dict)
                s_pubs = to_structured_data(raw_pubs, xpaths)

                for idx, p in enumerate(s_pubs):
                    s_pubs[idx]['authors'] = parser.parse_authors([a for a in p['authors']] if p['authors'] else [])
                    s_pubs[idx]['keywords'] = parser.parse_keywords(p['keywords'])
                    s_pubs[idx]['abstract'] = parser.parse_abstracts(p['abstract'])
                    s_pubs[idx]['body'] = parser.parse_bodies(p['body'])
                    s_pubs[idx]['doi'] = parser.parse_dois(p['doi'])
                    yield s_pubs[idx]


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
