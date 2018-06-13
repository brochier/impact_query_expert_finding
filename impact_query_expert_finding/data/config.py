import yaml

def load_from_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.load(f)
        return config