from training.utils.Config import toConfig
import json
import os

dirname = os.path.dirname(os.path.abspath(__file__))
jsons = [os.path.join(dirname, json_file) for json_file in os.listdir(dirname) if json_file.endswith('.json')]
configs = {os.path.basename(path)[:-5]: path for path in jsons }

def load(name):
    if name not in configs:
        raise KeyError(f"Not exists '{name}' Config")

    with open(configs[name], 'r') as f:
        config = toConfig(json.load(f))
        f.close()

    return config