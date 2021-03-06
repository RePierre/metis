import os
import gzip
import logging
import xmltodict as xd
from dictquery import DictQuery
from articleparser import ArticleParser


class FileParser():
    """Iterates over files in a specific directory and parses article information

    """

    def __init__(self, debug_mode):
        self._xpaths = [
            ('article_title', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/title-group/article-title'),
            ('journal_title', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/journal-title-group/journal-title'),
            ('doi', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/article-id'),
            ('publisher_name', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-name'),
            ('publisher_location', 'OAI-PMH/ListRecords/record/metadata/article/front/journal-meta/publisher/publisher-loc'),
            ('keywords', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/kwd-group'),
            ('abstract', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/abstract/p'),
            ('authors1', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/author-notes/fn'),
            ('authors2', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/contrib-group'),
            ('authors_affiliations', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/aff'),
            ('authors_affiliations2', 'OAI-PMH/ListRecords/record/metadata/article/front/article-meta/contrib-group/aff'),
            # ('acknowledgement', 'OAI-PMH/ListRecords/record/metadata/article/back/ack/p'),
            ('identifier', 'OAI-PMH/ListRecords/record/header/identifier'),
            ('body', 'OAI-PMH/ListRecords/record/metadata/article/body/sec')
        ]
        self._logger = logging.getLogger(__name__)
        self._debug_mode = debug_mode

    def parse_files(self, xml_path):
        raw_pubs = list()
        parser = ArticleParser()
        xml_sub_paths = [sp[0] for sp in os.walk(xml_path)]

        for sp in xml_sub_paths:
            for root, dirs, files in os.walk(sp):
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
                            s_pubs[idx]['authors'] = parser.parse_authors1([a for a in p['authors1']] if p['authors1'] else [])
                            s_pubs[idx]['authors'].extend(parser.parse_authors2(p['authors2']))
                            del s_pubs[idx]['authors1']  # remove temporary key
                            del s_pubs[idx]['authors2']  # remove temporary key
                            authors_affiliations = s_pubs[idx]['authors_affiliations'] if s_pubs[idx]['authors_affiliations'] else s_pubs[idx]['authors_affiliations2']
                            s_pubs[idx]['authors'] = parser.match_authors_affiliation(s_pubs[idx]['authors'], authors_affiliations)
                            del s_pubs[idx]['authors_affiliations']  # remove temporary key
                            del s_pubs[idx]['authors_affiliations2']  # remove temporary key
                            s_pubs[idx]['keywords'] = parser.parse_keywords(p['keywords'])
                            s_pubs[idx]['abstract'] = parser.parse_abstracts(p['abstract'])
                            s_pubs[idx]['body'] = parser.parse_bodies(p['body'])
                            s_pubs[idx]['doi'] = parser.parse_dois(p['doi'])
                            s_pubs[idx]['article_title'] = parser.parse_article_title(s_pubs[idx]['article_title'])
                            # print('idx: {}, authors: {}'.format(idx, s_pubs[idx]['authors']))
                            if s_pubs[idx]['authors'] and s_pubs[idx]['authors'][0]['affiliations'] and s_pubs[idx]['authors'][0]['affiliations'][0] and s_pubs[idx]['article_title'] and s_pubs[idx]['abstract'] and \
                                s_pubs[idx]['body'] and s_pubs[idx]['doi'] and s_pubs[idx]['keywords']:
                                yield s_pubs[idx]
                        # os.remove(file_path)

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
