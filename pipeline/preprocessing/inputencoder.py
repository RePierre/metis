import spacy
import numpy as np
import json
import os
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


def read_input(input_path, text_time_steps=2000,
               title_time_steps=100,
               keywords_time_steps=100):
    texts, affiliations, citations, keywords, titles = [], [], [], [], []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            filename = os.path.join(root, file)
            txt, aff, cit, kw, ttl = read_sample(filename)
            # Ingore file if any of the properties is None
            if not _is_valid_article(txt, aff, cit, kw, ttl):
                continue
            texts.append(txt)
            affiliations.append(aff)
            citations.append(cit)
            keywords.append(kw)
            titles.append(ttl)

    texts = pad_and_reshape(texts, text_time_steps)
    affiliations = np.reshape(affiliations, (len(affiliations), INPUT_SIZE))
    citations = np.reshape(citations, (len(citations), INPUT_SIZE))
    keywords = pad_and_reshape(keywords, keywords_time_steps)
    titles = pad_and_reshape(titles, title_time_steps)
    return texts, affiliations, citations, keywords, titles


def read_sample(filename):
    try:
        print('Reading file {}'.format(filename))
        with open(filename, 'rt') as f:
            data = json.load(f)
        text = encode_text(_build_text(data))
        affiliations = encode_affiliations(_build_affiliations(data))
        citations = encode_citations(_load_citations(data))
        keywords = encode_keywords(_build_keywords(data))
        title = encode_title(data['article_title'])
    except json.decoder.JSONDecodeError:
        return None, None, None, None, None
    except ValueError:
        return None, None, None, None, None
    return title, affiliations, keywords, text, citations


def _is_valid_article(*arrays):
    for arr in arrays:
        if arr is None:
            return False
        if type(arr) is list and len(arr) == 0:
            return False
        if type(arr) is np.ndarray and arr.size == 0:
            return False
    return True


def _build_keywords(data):
    keywords = []
    for kw in data['keywords']:
        if isinstance(kw, dict) and '#text' in kw:
            keywords.append(kw['#text'])
        if isinstance(kw, str):
            keywords.append(kw)

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
        title = section['title']
        if not isinstance(title, str):
            raise ValueError('Title is not a string.')

        contents = section['text']
        if not isinstance(contents, str):
            raise ValueError('Section content is not a string.')

        text.append(title)
        text.append(contents)
    return '\n'.join(text)


def _load_citations(data):
    return 0


def _load_affiliations():
    return {}
