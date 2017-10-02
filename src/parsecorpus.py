#!/usr/bin/env python3
import sys
import csv
import xml.etree.ElementTree as etree


def get_qualified_name(ns, tag, nsmap):
    url = nsmap[ns]
    name = '{' + url + '}' + tag
    return name


def get_article_info(article, nsmap):
    front = article.find(get_qualified_name('', 'front', nsmap))
    article_meta = front.find(get_qualified_name('', 'article-meta', nsmap))
    title_group = article_meta.find(get_qualified_name('', 'title-group', nsmap),
                                    namespaces=nsmap)
    for id in article_meta.findall(get_qualified_name('', 'article-id', nsmap),
                                   namespaces=nsmap):
        idtype = id.get('pub-id-type')
        for title in title_group.findall(get_qualified_name('', 'article-title', nsmap),
                                         namespaces=nsmap):
            yield (idtype, id.text, title.text)


def parse(xml):
    nsmap = {}
    context = etree.iterparse(xml, events=['end', 'start-ns'])
    for event, elem in context:
        if event == 'start-ns':
            ns, url = elem
            nsmap[ns] = url
        if event == 'end':
            if elem.tag == get_qualified_name('', 'article', nsmap):
                for tuple in get_article_info(elem, nsmap):
                    yield tuple
                elem.clear()


def main(inputfile, outputfile):
    with open(outputfile, 'wt') as f:
        writer = csv.writer(f, delimiter='\t')
        for item in parse(inputfile):
            writer.writerow(item)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
