import logging
from functools import reduce
from mongoengine import Document
from mongoengine import StringField
from mongoengine import ListField
from mongoengine import ReferenceField
from mongoengine import connect


class Author(Document):
    name = StringField(required=True)
    affiliations = ListField(StringField())


class Article(Document):
    doi = StringField(required=True, max_length=64)
    title = StringField(required=True, max_length=256)
    authors = ListField(ReferenceField(Author))
    keywords = ListField(StringField(max_length=128))
    abstract = StringField(required=True)
    text = StringField(required=True)


class DataStore:
    def __init__(self, db='pmc_oa',
                 host='localhost', port=27017):
        self._dbName = db
        self._port = port
        self._host = host
        self._logger = logging.getLogger(__name__)

    def store_publications(self, publications):
        connect(db=self._dbName,
                host=self._host,
                port=self._port)
        for _, pub in enumerate(publications):
            article = self._convert_to_article(pub)
            for author in self._convert_authors(pub):
                author.save()
                article.authors.append(author)
            self._log_article_save(article)
            article.save()

    def _convert_authors(self, pub):
        if not pub['authors']:
            return []
        return [Author(name=a['name'],
                       affiliations=a['affiliations'] if 'affiliations' in a else [])
                for _, a in enumerate(pub['authors'])
                if a is not None]

    def _convert_to_article(self, pub):
        article = Article()
        article.doi = pub['doi']
        article.title = self._get_text(pub['article_title'])
        article.keywords = self._convert_article_keywords(pub)
        article.abstract = pub['abstract']
        article.text = self._convert_article_text(pub)
        return article

    def _convert_article_keywords(self, pub):
        kwdlist = pub['keywords'] if 'keywords' in pub else []
        result = [self._get_text(kwd) for kwd in kwdlist]
        return result

    def _convert_article_text(self, pub):
        sections = [[self._get_text(sec['title']), self._get_text(sec['text'])]
                    for sec in pub['body']]
        flat = reduce(list.__add__, sections)
        text = '\n'.join(flat)
        return text

    def _get_text(self, item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if '#text' in item:
                return item['#text']
        return ''

    def _log_article_save(self, article):
        message = "Saving article {}.".format(article.doi)
        self._logger.info(message)
