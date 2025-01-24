import os

def get_data_path(filename):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', filename)
    return data_path

def data_path():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(base_path, 'data')
    return data_directory

def list_data_files():
    data_directory = data_path()
    try:
        files = os.listdir(data_directory)
        return [f for f in files if os.path.isfile(os.path.join(data_directory, f))]
    except FileNotFoundError:
        return []