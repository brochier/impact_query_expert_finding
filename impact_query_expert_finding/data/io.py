import os
import json
import shutil
import numpy as np
import yaml

def save_as_json(path, name, data_to_save):
    check_and_create_dir(path)
    data_to_save = serialize_for_json(data_to_save)
    file_path = os.path.join(path, name)
    print("Saving as json to: ", file_path)
    with open(file_path, 'w') as outfile:
        json.dump(data_to_save, outfile)

def load_as_json(path, name):
    file_path = os.path.join(path, name)
    print("Load as json from: ", file_path)
    with open(file_path, 'r') as infile:
        data_loaded = json.load(infile)
    return data_loaded

def save_as_yaml(path, name, data_to_save):
    check_and_create_dir(path)
    file_path = os.path.join(path, name)
    print("Saving as yaml to: ", file_path)
    with open(file_path, 'w') as outfile:
        config = yaml.dump(data_to_save, outfile)
        return config

def load_as_yaml(path, name):
    file_path = os.path.join(path, name)
    print("Load as yaml from: ", file_path)
    with open(file_path, 'r') as infile:
        data_loaded = yaml.load(infile)
    return data_loaded


def serialize_for_json(data_to_serialize):
    if isinstance(data_to_serialize, set):
        print("Data saved is being converted from set() to list() data structure for JSON serialiszation !")
        return list(data_to_serialize)
    elif isinstance(data_to_serialize, np.ndarray):
        print("Data saved is being converted from np.ndarray() to list() data structure for JSON serialiszation !")
        return data_to_serialize.tolist()
    else:
        return data_to_serialize

def copy_past_dir(src, dest):
    print("Updating files from ", src, " to ", dest)
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def get_extension(file_path):
    return file_path.split(".")[-1]

def list_dir(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles

def check_and_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Path did not exist. Creating path ", path)
        return False
    return True

def check_dir(path):
    if not os.path.isdir(path):
        return False
    return True