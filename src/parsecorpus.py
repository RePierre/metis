#!/usr/bin/env python3
import sys
import csv
from pprint import pprint
import xml.etree.ElementTree as etree


class ArticleInfoParser():
    """Parses article info
    """

    def __init__(self):
        self._nsmap = {}
        self._parseevents = ['start-ns', 'start', 'end']

    def _get_qname(self, nsprefix, tag):
        url = self._nsmap[nsprefix]
        name = '{' + url + '}' + tag
        return name

    def _isarticle(self, elem):
        return elem.tag == self._get_qname('', 'article')

    def _isarticleid(self, elem):
        return elem.tag == self._get_qname('', 'article-id')

    def _parsearticleid(self, elem):
        idtype = elem.get('pub-id-type')
        id = elem.text
        return (id, idtype)

    def _istitle(self, elem):
        return elem.tag == self._get_qname('', 'article-title')

    def _parsetitle(self, elem):
        text = ''.join(elem.itertext())
        return text

    def parse(self, xml):
        context = etree.iterparse(xml, events=self._parseevents)
        for event, elem in context:
            if event == 'start-ns':
                ns, url = elem
                self._nsmap[ns] = url
            if event == 'start':
                if self._isarticle(elem):
                    info = {'ids': []}
            if event == 'end':
                if self._isarticleid(elem):
                    info['ids'].append(self._parsearticleid(elem))
                if self._istitle(elem):
                    info['title'] = self._parsetitle(elem)
                if self._isarticle(elem):
                    yield info
                    elem.clear()


if __name__ == '__main__':
    inputfile = sys.argv[1]
    # outputfile = sys.argv[2]
    parser = ArticleInfoParser()
    for d in parser.parse(inputfile):
        pprint(d)
