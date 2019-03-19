import os
import json
import sys
import numpy as np


class Corpus:
    """
    Represents a corpus directory as an iterable.
    """

    def __init__(self, corpus_dir,
                 num_files=None,
                 num_sections=None,
                 shuffle_files=True,
                 logger=None):
        """
        Initializes an instance of a Corpus class.

        Parameters
        ----------
        corpus_dir: string
            The full path to the corpus directory.
        num_files: int, optional
            If provided, the corpus will read only `num_files` files; other files will be ignored.
            Default is `None`, in which case all files will be read.
        num_sections: int, optional
            If provided, only the first `num_sections` of each file will be returned.
            Default is `None`, in which case all sections will be returned.
        shuffle_files: bool, optional
            Specifies whether to read the files in order or shuffle them before reading.
            Default is `True`.
        logger: logger, optional
            If provided will be used to write debug messages.
            Default is `None`, in which case no logging will be outputted.
        """
        self._corpus_dir = corpus_dir
        self._num_files = num_files
        self._num_sections = num_sections
        self._shuffle_files = shuffle_files
        self._logger = logger

    def __iter__(self):
        self._log_debug_message('Start iterating over corpus directory {}.'
                                .format(self._corpus_dir))
        self._log_debug_message('Current configuration: num_files: {}, num_sections: {}, shuffle_files: {}'
                                .format(self._num_files, self._num_sections, self._shuffle_files))
        files = [f for f in self._get_file_names()]
        if self._shuffle_files:
            np.random.shuffle(files)
        max_files = self._num_files if self._num_files else sys.maxsize
        index = 0
        for f in files:
            if index < max_files:
                try:
                    yield self._get_file_text(f)
                    index +=1
                except json.decoder.JSONDecodeError:
                    continue
                except TypeError:
                    continue
            else:
                break
        self._log_debug_message('Finished iterating over corpus directory {}.'
                                .format(self._corpus_dir))

    def _log_debug_message(self, message):
        if self._logger:
            self._logger.debug(message)

    def _get_file_names(self):
        for root, dirs, files in os.walk(self._corpus_dir):
            for f in files:
                yield os.path.join(root, f)

    def _get_file_text(self, file_name):
        text = []
        max_sections = self._num_sections if self._num_sections else sys.maxsize
        with open(file_name, 'rt') as f:
            data = json.load(f)
            for index, section in enumerate(data['body']):
                if index < max_sections:
                    text.append(section['title'])
                    text.append(section['text'])
                else:
                    break
            return '\n'.join(text)
