from inputencoder import read_input
from sklearn.model_selection import train_test_split


def build_datasets(input_path, batch_size):
    texts, affiliations, citations, keywords, titles = read_input(
        input_path)
    # Make sure the length of collections is even
    if len(texts) % 2 != 0:
        texts, affiliations, citations, keywords, titles = _drop_last_item(
            texts, affiliations, citations, keywords, titles)
    # Split the data in half to accomodate for 2 inputs of the network
    txt1, txt2, aff1, aff2, cit1, cit2, kwd1, kwd2, ttl1, ttl2 = train_test_split(
        texts, affiliations, citations, keywords, titles,
        train_size=0.5, shuffle=False)
    txt1_train, txt1_test, \
        aff1_train, aff1_test, \
        cit1_train, cit1_test, \
        kwd1_train, kwd1_test, \
        ttl1_train, ttl1_test, \
        txt2_train, txt2_test, \
        aff2_train, aff2_test, \
        cit2_train, cit2_test, \
        kwd2_train, kwd2_test, \
        ttl2_train, ttl2_test = _accomodate_batch_size(
            batch_size,
            train_test_split(txt1, aff1, cit1, kwd1, ttl1, txt2, aff2, cit2, kwd2, ttl2))
    input1_train = InputData(txt1_train, aff1_train, cit1_train, kwd1_train, ttl1_train)
    input2_train = InputData(txt2_train, aff2_train, cit2_train, kwd2_train, ttl2_train)
    input1_test = InputData(txt1_test, aff1_test, cit1_test, kwd1_test, ttl1_test)
    input2_test = InputData(txt2_test, aff2_test, cit2_test, kwd2_test, ttl2_test)
    return input1_train, input2_train, input1_test, input2_test


def _accomodate_batch_size(batch_size, *arrays):
    result = []
    for arr in arrays:
        num_samples = int(len(arr) / batch_size) * batch_size
        result.append(arr[:num_samples])
    return result


def _drop_last_item(*arrays):
    return [a[:-1] for a in arrays]


class InputData():
    def __init__(self, texts, affiliations, citations, keywords, titles):
        self._texts = texts
        self._affiliations = affiliations
        self._citations = citations
        self._keywords = keywords
        self._titles = titles

    @property
    def texts(self):
        return self._texts

    @property
    def affiliations(self):
        return self._affiliations

    @property
    def citations(self):
        return self._citations

    @property
    def keywords(self):
        return self._keywords

    @property
    def titles(self):
        return self._titles
