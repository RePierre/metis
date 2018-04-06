import spacy
import numpy as np
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
    return encode_title(keywords)


def encode_citations(num_citations):
    return np.full((INPUT_SIZE,), num_citations)


def encode_affiliation(affiliation):
    aff = _load_affiliations()
    affiliation = affiliation.lower()
    affiliation_id = aff[affiliation] if affiliation in aff else -1
    return np.full((INPUT_SIZE, 1), affiliation_id)


def _load_affiliations():
    return {}
