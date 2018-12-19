from corpusutils import read_corpus, get_file_names, get_file_text
import numpy as np
import pdb


class FixedSizeCorpus:
    def __init__(self, corpus_dir, num_files,
                 random_order=True, logger=None):
        self._corpus_dir = corpus_dir
        self._num_files = num_files
        self._random_order = random_order
        self._logger = logger

    def __iter__(self):
        if not self._random_order:
            num_files = self._num_files
            for text in read_corpus(self._corpus_dir):
                if num_files > 0:
                    num_files -= 1
                    yield text
        else:
            files = [f for f in get_file_names(self._corpus_dir)]
            np.random.shuffle(files)
            for f in files[:self._num_files]:
                yield get_file_text(f)

    def _log_debug_message(self, message):
        if self._logger:
            self._logger.debug(message)
