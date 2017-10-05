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
            for it in items:
                a_info = dict()
                name_items = it.strip().split(',')
                a_info['name']        = name_items[0].strip()
                a_info['affiliation'] = name_items[1].strip()
                a_info['country']     = name_items[2].strip()
                res.append(a_info)
    return res


def parse_keywords(keywords):
    res = []
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


def parse_files(xml_path):

    xpaths = [
        ('article_title',      'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/title-group/article-title'),
        ('journal_title',      'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/journal-title-group/journal-title'),
        ('publisher_name',     'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-name'),
        ('publisher_location', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-loc'),
        ('keywords',           'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/kwd-group'),
        ('abstract',           'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/abstract/p'),
        ('authors',            'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/author-notes/fn'),
        ('acknowledgement',    'OAI-PMH/ListRecords/record/metadata/article/back/ack/p'),
        ('identifier',         'OAI-PMH/ListRecords/record/header/identifier'),
        ('body',               'OAI-PMH/ListRecords/record/metadata/article/body/sec')
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
                s_pubs = to_structured_data(raw_pubs, xpaths)

                for idx, p in enumerate(s_pubs):
                    s_pubs[idx]['authors']  = parse_authors([a for a in p['authors']] if p['authors'] else [])
                    s_pubs[idx]['keywords'] = parse_keywords(p['keywords'])
                    s_pubs[idx]['abstract'] = parse_abstracts(p['abstract'])
                    s_pubs[idx]['body']     = parse_bodies(p['body'])
                    yield s_pubs[idx]


def store_output(pubs, output_path):
    for idx, pub in enumerate(pubs):
        with open(os.path.join(output_path, str(idx)+'.json'), 'w') as f:
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
