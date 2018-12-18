import os
import json


def get_file_names(corpus_dir):
    for root, dirs, files in os.walk(corpus_dir):
        for f in files:
            yield os.path.join(root, f)


def get_file_text(file_name):
    text = []
    with open(file_name, 'rt') as f:
        data = json.load(f)
        for section in data['body']:
            text.append(section['title'])
            text.append(section['text'])
        return '\n'.join(text)


def read_corpus(corpus_dir):
    for file_name in get_file_names(corpus_dir):
        try:
            yield get_file_text(file_name)
        except json.decoder.JSONDecodeError:
            continue
        except TypeError:
            continue
