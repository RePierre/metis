#!/usr/bin/env python3
import sys
import csv
from pprint import pprint
import xml.etree.ElementTree as etree


class ArticleSection():
    """Represents an article section

    """

    def __init__(self):
        self._title = None
        self._text = None

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        value = value.strip()
        if self.title is not None:
            self._text = value.replace(self.title, '', 1)
        else:
            self._text = value

    def asdictionary(self):
        return {'title': self.title if self.title is not None else '',
                'text': self.text}


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

    def _is(self, elem, name, ns=''):
        return elem.tag == self._get_qname(ns, name)

    def _isarticle(self, elem):
        return self._is(elem, 'article')

    def _isarticleid(self, elem):
        return self._is(elem, 'article-id')

    def _parsearticleid(self, elem):
        idtype = elem.get('pub-id-type')
        id = elem.text
        return (id, idtype)

    def _istitle(self, elem):
        return self._is(elem, 'article-title')

    def _parsetext(self, elem, strip=False):
        text = ''.join(elem.itertext())
        return text.strip() if strip else text

    def _isabstract(self, elem):
        return self._is(elem, 'abstract')

    def _issection(self, elem):
        return self._is(elem, 'sec')

    def _issectiontitle(self, elem):
        return self._is(elem, 'title')

    def parse(self, xml):
        context = etree.iterparse(xml, events=self._parseevents)
        for event, elem in context:
            if event == 'start-ns':
                ns, url = elem
                self._nsmap[ns] = url
            if event == 'start':
                if self._isarticle(elem):
                    info = {'ids': [],
                            'sections': []}
                if self._issection(elem):
                    section = ArticleSection()
            if event == 'end':
                if self._isarticleid(elem):
                    info['ids'].append(self._parsearticleid(elem))
                if self._istitle(elem):
                    info['title'] = self._parsetext(elem)
                if self._isabstract(elem):
                    info['abstract'] = self._parsetext(elem, strip=True)
                if self._issectiontitle(elem):
                    section.title = self._parsetext(elem)
                if self._issection(elem):
                    section.text = self._parsetext(elem, strip=True)
                    info['sections'].append(section.asdictionary())
                if self._isarticle(elem):
                    yield info
                    elem.clear()


if __name__ == '__main__':
    inputfile = sys.argv[1]
    # outputfile = sys.argv[2]
    parser = ArticleInfoParser()
    for d in parser.parse(inputfile):
        pprint(d)
