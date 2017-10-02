import os
from lxml import etree
import gzip
import argparse
import logging
from io import StringIO, BytesIO
import xmltodict as xd

LOG = logging.getLogger(__name__)


# taken from https://www.haykranen.nl/2016/02/13/handling-complex-nested-dicts-in-python/
class DictQuery(dict):
    def get(self, path, default=None):
        keys = path.split("/")
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)

            if not val:
                break

        return val


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
        ('article_title',      'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/title-group/article-title'),
        ('journal_title',      'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/journal-title-group/journal-title'),
        ('publisher_name',     'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-name'),
        ('publisher_location', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-loc')
    ]

    raw_pubs = list()

    for root, dirs, files in os.walk(xml_path):
        for file in files:
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                LOG.info('Parsing {}'.format(file_path))
                with gzip.open(file_path, 'r') as gzip_file:
                    pub_xml = gzip_file.read()
                    pub_dict = xd.parse(pub_xml)
                    raw_pubs.append(pub_dict)
                    # context = etree.iterparse(BytesIO(pub_xml))
                    # pub_dict = dict()
                    # for action, elem in context:
                    #     text = elem.text
                    #     # print(elem.tag)
                    #     # print(elem.tag + " => " + text)
                    #     if "journal-title" in elem.tag:
                    #         pub_dict[elem.tag] = text
                    #         pubs.append(pub_dict)
                    #         pub_dict = {}
    s_pubs = to_structured_data(raw_pubs, xpaths)
    return s_pubs


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--xml_path', help="The URL to XML folder", required=True)
    args = argparser.parse_args()

    # check xml path exists
    assert os.path.exists(args.xml_path), '{} does not exist!'.format(args.xml_path)

    # do the actual parsing
    s_data = parse_files(args.xml_path)

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
