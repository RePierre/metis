from corpusutils import read_corpus


class FixedSizeCorpus:
    def __init__(self, corpus_dir, num_files,
                 random_order=True, logger=None):
        self._corpus_dir = corpus_dir
        self._num_files = num_files
        self._random_order = random_order
        self._logger = logger

    def __iter__(self):
        if not self._random_order:
            read_corpus(self._corpus_dir)

    def _log_debug_message(self, message):
        if self._logger:
            self._logger.debug(message)
