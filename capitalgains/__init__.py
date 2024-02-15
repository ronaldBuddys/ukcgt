import os

def get_path(*sub_dir):
    """get_path to package"""
    return os.path.join(os.path.dirname(__file__), *sub_dir)

def get_data_path(*sub_dir):
    return get_path('data', *sub_dir)
