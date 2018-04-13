import spacy
import numpy as np
import json
from keras.preprocessing.sequence import pad_sequences

INPUT_SIZE = 384
TEXT_SIZE = 5000
nlp = spacy.load('en')


def pad_and_reshape(sequence, time_steps):
    sequence = pad_sequences(sequence, maxlen=time_steps, dtype='float32')
    num_samples, num_features = sequence.shape[0], INPUT_SIZE
    sequence = np.reshape(sequence, (num_samples, time_steps, num_features))
    return sequence


def encode_text(text):
    tokens = [t.vector for t in nlp(text.lower())
              if not t.is_stop and not t.is_punct]
    return tokens


def encode_title(title):
    return encode_text(title)


def encode_keywords(keywords):
    return encode_text(keywords)


def encode_citations(num_citations):
    return np.full((INPUT_SIZE,), num_citations)


def encode_affiliations(affiliations):
    aff = _load_affiliations()
    encoded_affiliations = np.full((INPUT_SIZE, 1), -1)
    for index, affiliation in enumerate(affiliations):
        affiliation = affiliation.lower()
        affiliation_id = aff[affiliation] if affiliation in aff else -1
        encoded_affiliations[index] = affiliation_id
    return encoded_affiliations


def read_sample(filename, text_time_steps=2000,
                title_time_steps=100, keywords_time_steps=100):
    with open(filename, 'rt') as f:
        data = json.load(f)
    text = pad_and_reshape(
        encode_text(_build_text(data)),
        text_time_steps)
    affiliations = encode_affiliations(_build_affiliations(data))
    citations = encode_citations(_load_citations(data))
    keywords = pad_and_reshape(
        encode_keywords(_build_keywords(data)),
        keywords_time_steps)
    title = pad_and_reshape(
        encode_title(data['article_title']),
        title_time_steps)
    return title, affiliations, keywords, text, citations


def _build_keywords(data):
    keywords = []
    for kw in data['keywords']:
        keywords.append(kw['#text'])
    return ' '.join(keywords)


def _build_affiliations(data):
    affiliations = []
    for author in data['authors']:
        for affiliation in author['affiliations']:
            affiliations.append(affiliation)
    return affiliations


def _build_text(data):
    text = []
    for section in data['body']:
        text.append(section['title'])
        text.append(section['text'])
    return '\n'.join(text)


def _load_citations(data):
    return 0


def _load_affiliations():
    return {}
