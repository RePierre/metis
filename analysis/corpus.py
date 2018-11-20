import os
import json


class Corpus:
    def __init__(self, corpus_dir):
        self._corpus_dir = corpus_dir

    def __iter__(self):
        for file_name in self._get_file_names():
            try:
                yield self._get_file_text(file_name)
            except json.decoder.JSONDecodeError:
                continue
            except TypeError:
                continue

    def _get_file_names(self):
        for root, dirs, files in os.walk(self._corpus_dir):
            for f in files:
                yield os.path.join(root, f)

    def _get_file_text(self, file_name):
        text = []
        with open(file_name, 'rt') as f:
            data = json.load(f)
            for section in data['body']:
                text.append(section['title'])
                text.append(section['text'])
            return '\n'.join(text)
