import os
import json
import collections
import gzip
import argparse
import logging
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
    """
    Convert raw dict to an easy to parse form
    :param raw_pubs: list of raw pub dicts.
    :param xpaths: list of xpath-like string to select the needed information.
    :return: list of easier to parse dictionaries.
    """
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


def parse_authors(authors):
    res = list()
    for a in authors:
        if 'sup' not in a['p'] and 'italic' in a['p']:
            items = a['p']['italic'].split(';')
            for author_item in items:
                a_info = dict()
                a_info['raw'] = author_item.strip()
                name_items    = author_item.strip().split(',')
                a_info['name']        = name_items[0].strip()
                a_info['affiliation'] = name_items[1].strip()
                a_info['country']     = name_items[2].strip()
                res.append(a_info)
    return res


def parse_keywords(keywords):
    """
    Convert raw list of keywords to a cleaned-up version.
    :param keywords: list of raw keywords
    :return: list of cleaned-up keywords
    """
    res = list()
    if keywords:
        if 'kwd' in keywords:
            res.extend(keywords['kwd'])
        else:
            for item in keywords:
                if 'title' in item and 'kwd' in item:
                    res.extend(item['kwd'])
    return res


def parse_abstracts(abstract):
    res = ''
    if abstract:
        if '#text' in abstract:
            res = abstract['#text']
        elif type(abstract) == list:
            for item in abstract:
                if 'bold' in item:
                    res += '{} {}'.format(item['bold'], item['#text'])
                elif '#text' in item:
                    res += item['#text']
                else:
                    res += item
        else:
            res = abstract
    return res


def parse_bodies(body):
    res = list()
    if body:
        for item in body:
            sec_d = dict()
            if type(item) == collections.OrderedDict:
                sec_d['title'] = item['title']
                sec_d['text'] = ''
                if 'p' in item:
                    if type(item) == collections.OrderedDict:
                        if type(item['p']) == collections.OrderedDict:
                            for k, v in item['p'].items():
                                if k == '#text':
                                    sec_d['text'] += v
                        elif type(item['p']) == list:
                            for it in item['p']:
                                if '#text' in it:
                                    sec_d['text'] += it['#text']
                        elif type(item['p']) == str:
                            sec_d['text'] += item['p']
                    elif type(item) == list:
                        for it in item:
                            if '#text' in item:
                                sec_d['text'] += it['#text']
                elif 'sec' in item:
                    for sec_item in item['sec']:
                        if type(sec_item) == collections.OrderedDict and sec_item.get('p'):
                            if '#text' in sec_item['p']:
                                sec_d['text'] += sec_item['p']['#text']
                            elif type(sec_item['p']) == list:
                                for it in sec_item['p']:
                                    if '#text' in it:
                                        sec_d['text'] += it['#text']
                            elif type(sec_item['p']) == str:
                                sec_d['text'] += sec_item['p']
                        elif type(sec_item) == list:
                            for item in sec_item:
                                if '#text' in item:
                                    sec_d['text'] += item['#text']
            if sec_d:
                res.append(sec_d)
    return res


def parse_ids(identif_dict):
    res = dict()
    for idd in identif_dict:
        if idd['@pub-id-type'] == 'doi':
            res['doi'] = idd['#text']
        elif idd['@pub-id-type'] == 'pmcid':
            res['pmcid'] = idd['#text']
        elif idd['@pub-id-type'] == 'pmc-uid':
            res['pmc-uid'] = idd['#text']
    return res


def parse_files(xml_path):
    """
    Parse all .gz files to .json from a given folder
    :param xml_path: path where .gz files are located
    :return: Publication dictionary
    """

    xpaths = [
        ('article_title',      'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/title-group/article-title'),
        ('journal_title',      'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/journal-title-group/journal-title'),
        ('ids',                'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/article-id'),
        ('publisher_name',     'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-name'),
        ('publisher_location', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-loc'),
        ('keywords',           'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/kwd-group'),
        ('abstract',           'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/abstract/p'),
        ('authors',            'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/author-notes/fn'),
        ('acknowledgement',    'OAI-PMH/ListRecords/record/metadata/article/back/ack/p'),
        ('identifier',         'OAI-PMH/ListRecords/record/header/identifier'),
        ('body',               'OAI-PMH/ListRecords/record/metadata/article/body/sec'),
    ]

    raw_pubs = list()

    # iterate through all the files
    for root, dirs, files in os.walk(xml_path):
        for file in files:

            # select those with wanted extension
            if file.endswith(".gz"):
                file_path = os.path.join(root, file)
                LOG.info('Parsing {}'.format(file_path))
                with gzip.open(file_path, 'r') as gzip_file:
                    pub_xml = gzip_file.read()
                    pub_dict = xd.parse(pub_xml)
                    raw_pubs.append(pub_dict)

                # convert raw dict to an easy to parse form
                s_pubs = to_structured_data(raw_pubs, xpaths)

                # do post-processing to convert raw data in a nice, easy-to-use format
                for idx, p in enumerate(s_pubs):
                    s_pubs[idx]['authors']  = parse_authors([a for a in p['authors']] if p['authors'] else [])
                    s_pubs[idx]['keywords'] = parse_keywords(p['keywords'])
                    s_pubs[idx]['abstract'] = parse_abstracts(p['abstract'])
                    s_pubs[idx]['body']     = parse_bodies(p['body'])
                    ids                    = parse_ids(p['ids'])
                    s_pubs[idx].pop('ids', None)  # remove attribute when no longer needed
                    s_pubs[idx]['doi']     = ids.get('doi')
                    s_pubs[idx]['pmcid']   = ids.get('pmcid')
                    s_pubs[idx]['pmc-uid'] = ids.get('pmc-uid')

                    # it's a generator to reduce memory consumption
                    yield s_pubs[idx]


def store_output(pubs, output_path):
    for idx, pub in enumerate(pubs):
        assert 'pmcid' in pub, 'Pub is missing key'
        file_path = os.path.join(output_path, '{}.json'.format(pub['pmcid']))
        LOG.info('Storing output to file {}'.format(file_path))
        with open(file_path, 'w') as f:
            f.write(json.dumps(pub, indent=True, sort_keys=True, ensure_ascii=False))


def run():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s:%(name)s %(funcName)s: %(message)s')

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path',  help="The URL to XML input folder",   required=True)
    argparser.add_argument('--output_path', help="The URL to JSON output folder", required=True)
    args = argparser.parse_args()

    # check paths exist
    assert os.path.exists(args.input_path),  '{} does not exist!'.format(args.input_path)
    assert os.path.exists(args.output_path), '{} does not exist!'.format(args.output_path)

    # do the actual parsing
    pubs = parse_files(args.input_path)
    store_output(pubs, args.output_path)

    LOG.info("That's all folks!")


if __name__ == "__main__":
    run()
