import collections


class ArticleParser():
    """Parses article information

    """

    def parse_authors1(self, authors):
        res = list()
        for a in authors:
            if type(a) == dict and 'sup' not in a['p'] and 'italic' in a['p']:
                items = a['p']['italic'].split(';')
                for author_item in items:
                    a_info = dict()
                    a_info['raw'] = author_item.strip()
                    name_items = author_item.strip().split(',')
                    a_info['name'] = name_items[0].strip()
                    a_info['affiliation'] = name_items[1].strip()
                    a_info['country'] = name_items[2].strip()
                    res.append(a_info)
        return res

    def get_names(self, item):
        if type(item) == collections.OrderedDict and item['@contrib-type'] == 'author':
            if 'xref' in item:
                affiliation_idx = [afidx['@rid'] for afidx in item['xref']] if type(item['xref']) == list else [item['xref']['@rid']]
            else:
                affiliation_idx = []
            return {'name': '{} {}'.format(item['name']['given-names'], item['name']['surname']), 'affiliation_idx': affiliation_idx}
        else:
            return None

    def parse_authors2(self, authors):
        res = list()
        if not authors:
            return res

        _authors = list()
        if type(authors) == list:
            for a in authors:
                for it in a.get('contrib'):
                    _authors.append(it)
        else:
            _authors = authors.get('contrib')
        for a in _authors:
            res.append(self.get_names(a))
        return res

    def match_authors_affiliation(self, authors, affiliations):
        res = authors.copy()
        if affiliations:
            _aff = self.build_affiliations(affiliations)
            for a in authors:
                if not a:
                    continue
                a['affiliations'] = []
                for af_idx in a['affiliation_idx']:
                    if af_idx in _aff:
                        a['affiliations'].append(_aff[af_idx])
                del a['affiliation_idx']  # remove temporary key
                res.append(a)
        return res

    def build_affiliations(self, affiliations):
        if type(affiliations) == list:
            return {it['@id']: it['#text'] if '#text' in it else ''
                    for it in affiliations}
        if isinstance(affiliations, dict) and '@id' in affiliations:
            return {affiliations['@id']:
                    affiliations['#text'] if '#text' in affiliations else ''}
        return {}

    def parse_keywords(self, keywords):
        res = []
        if keywords:
            if 'kwd' in keywords:
                res.extend(keywords['kwd'])
            else:
                for item in keywords:
                    if 'title' in item and 'kwd' in item:
                        res.extend(item['kwd'])
        return res

    def parse_abstracts(self, abstract):
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
                        res += item if isinstance(item, str) else ''
            else:
                res = abstract
        return res

    def parse_bodies(self, body):
        res = list()
        if body:
            for item in body:
                sec_d = dict()
                if type(item) == collections.OrderedDict:
                    sec_d['title'] = item['title'] if 'title' in item else ''
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

    def parse_dois(self, identif_dict):
        res = ''
        for idd in identif_dict:
            if idd['@pub-id-type'] == 'doi':
                res = idd['#text']
        return res
