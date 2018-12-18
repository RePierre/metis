from corpusutils import read_corpus


class Corpus:
    def __init__(self, corpus_dir, logger=None):
        self._corpus_dir = corpus_dir
        self._logger = logger

    def __iter__(self):
        self._log_debug_message('Start iterating over corpus directory {}.'
                                .format(self._corpus_dir))
        for text in read_corpus(self._corpus_dir):
            yield text
        self._log_debug_message('Finished iterating over corpus directory {}.'
                                .format(self._corpus_dir))

    def _log_debug_message(self, message):
        if self._logger:
            self._logger.debug(message)
