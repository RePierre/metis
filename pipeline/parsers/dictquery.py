class DictQuery(dict):
    """Handles nested dictionaries
    Taken from https://www.haykranen.nl/2016/02/13/handling-complex-nested-dicts-in-python/
    """

    def get(self, path, default=None):
        keys = path.split("/")
        val = dict.get(self, keys[0], default)

        for key in keys[1:]:
            val = self._get_recursive(key, val, default)
            if not val:
                break

        return val

    def _get_recursive(self, key, val, default):
        if not val:
            return None
        if isinstance(val, dict):
            return val.get(key, default)
        if isinstance(val, list):
            return [self._get_recursive(key, v, default) if v else None for v in val]
        return None
