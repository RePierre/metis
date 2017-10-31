class DictQuery(dict):
    """Handles nested dictionaries
    Taken from https://www.haykranen.nl/2016/02/13/handling-complex-nested-dicts-in-python/
    """

    def get(self, path, default=None):
        keys = path.split("/")
        val = None

        for key in keys:
            if val:
                if isinstance(val, list):
                    val = [v.get(key, default) if v else None for v in val]
                else:
                    val = val.get(key, default)
            else:
                val = dict.get(self, key, default)

            if not val:
                break

        return val
