import os
import gzip
import xmltodict as xd
from dictquery import DictQuery
from articleparser import ArticleParser


class FileParser():
    """Iterates over files in a specific directory and parses article information

    """

    def __init__(self, logger):
        self._xpaths = [
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
        self._logger = logger

    def parse_files(self, xml_path):
        raw_pubs = list()
        parser = ArticleParser()
        for root, dirs, files in os.walk(xml_path):
            for file in files:
                if file.endswith(".gz"):
                    file_path = os.path.join(root, file)
                    self._logger.info('Parsing {}'.format(file_path))
                    with gzip.open(file_path, 'r') as gzip_file:
                        pub_xml = gzip_file.read()
                        pub_dict = xd.parse(pub_xml)
                        raw_pubs.append(pub_dict)
                    s_pubs = self._to_structured_data(raw_pubs)

                    for idx, p in enumerate(s_pubs):
                        s_pubs[idx]['authors'] = parser.parse_authors([a for a in p['authors']] if p['authors'] else [])
                        s_pubs[idx]['keywords'] = parser.parse_keywords(p['keywords'])
                        s_pubs[idx]['abstract'] = parser.parse_abstracts(p['abstract'])
                        s_pubs[idx]['body'] = parser.parse_bodies(p['body'])
                        s_pubs[idx]['doi'] = parser.parse_dois(p['doi'])
                        yield s_pubs[idx]

    def _to_structured_data(self, raw_pubs):
        pubs = list()
        for rp in raw_pubs:
            s_pub = dict()
            # query for paths
            for key, path in self._xpaths:
                query_result = DictQuery(rp)
                s_pub[key] = query_result.get(path)
            for v_item in zip(*[s_pub[k] for k in s_pub.keys()]):
                tmp_dict = dict()
                for idx, k in enumerate(s_pub.keys()):
                    tmp_dict[k] = v_item[idx]
                pubs.append(tmp_dict)
        return pubs
