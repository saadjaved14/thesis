import cPickle as pickle
import json

import yaml


def pickle_load_file(p_file):
    with open(p_file, 'rb') as handle:
        data = pickle.load(handle)
    return data


def pickle_save_file(p_file, data):
    with open(p_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def yaml_save_file(p_filepath, p_data):
    """ Saves yaml file to the specified path"""
    with open(p_filepath, 'w') as handle:
        yaml.dump(p_data, handle, default_flow_style=False)


def yaml_load_file(p_filepath):
    """ Loads yaml file and returns a dict"""
    with open(p_filepath, 'r') as handle:
        data = yaml.load(handle)
    return data


def json_save_file(p_filepath, p_data):
    with open(p_filepath, 'w') as handle:
        json.dump(p_data, handle, sort_keys=True, indent=4)


def json_load_file(p_filepath):
    with open(p_filepath, 'r') as handle:
        data = json.load(handle)
    return data
