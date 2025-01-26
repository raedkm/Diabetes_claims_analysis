import os

def get_base_path():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return base_path

def get_data_path(filename):
    base_path = get_base_path()
    data_path = os.path.join(base_path, 'data', filename)
    return data_path

def data_path():
    base_path = get_base_path()
    data_directory = os.path.join(base_path, 'data')
    return data_directory

def list_data_files():
    data_directory = data_path()
    try:
        files = []
        for root, _, filenames in os.walk(data_directory):
            for filename in filenames:
                files.append(os.path.relpath(os.path.join(root, filename), data_directory))
        return files
    except FileNotFoundError:
        return []
    

def load_and_clean_data(filename):
    data_path = get_data_path(filename)
    df = pd.read_csv(data_path)
    
    # Example cleaning steps
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)
    
    return df