import json

def toConfig(src):
    if type(src) != dict:
        return src

    dst = Config()
    for k, v in src.items():
        if type(v) == dict:
            dst[k] = toConfig(v)
        elif type(v) == list:
            dst[k] = [toConfig(item) for item in v]
        else:
            dst[k] = v
    return dst

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            data = json.load()
            f.close()

        return toConfig()