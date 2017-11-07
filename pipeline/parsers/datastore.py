from mongoengine import Document
from mongoengine import StringField
from mongoengine import ListField
from mongoengine import ReferenceField
from mongoengine import connect


class Author(Document):
    name = StringField(required=True, max_length=64)
    affiliation = StringField(max_length=64)
    country = StringField(max_length=128)


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

    def store_publications(self, publications):
        connect(db=self._dbName,
                host=self._host,
                port=self._port)
        for _, pub in enumerate(publications):
            article = self._convert_to_article(pub)
            for author in self._convert_authors(pub):
                author.save()
                article.authors.append(author)
            article.save()

    def _convert_authors(self, pub):
        return [Author(name=a['name'],
                       affiliation=a['affiliation'],
                       country=a['country'])
                for _, a in enumerate(pub['authors'])]

    def _convert_to_article(self, pub):
        article = Article()
        article.doi = pub['doi']
        article.title = pub['title']
        article.keywords = pub['keywords']
        article.abstract = pub['abstract']
        article.text = pub['text']
        return article