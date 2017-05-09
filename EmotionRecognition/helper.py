import collections
import os
import filemanager as fm


def nested_dict():
    return collections.defaultdict(nested_dict)


def load_setup():
    # Load setup.yml from root folder
    d_setup = fm.yaml_load_file('setup.yml')
    if os.path.isfile(d_setup['classifierConfigPath']):
        d_config = fm.yaml_load_file(d_setup['classifierConfigPath'])
    else:
        d_config = {}
    return d_setup, d_config


def create_directory(pv_path):
    try:
        os.makedirs(pv_path)
    except OSError:
        if not os.path.isdir(pv_path):
            raise
