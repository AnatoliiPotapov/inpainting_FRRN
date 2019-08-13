import os
import yaml

def get_config(path):
    with open(path, 'r') as f:
        pretty_config = f.read()
        config = yaml.load(pretty_config, yaml.Loader)
    return config, pretty_config