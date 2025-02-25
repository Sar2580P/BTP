import zipfile
import os
from omegaconf import OmegaConf

def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

def unzip_file(zip_path, extract_to):
    """
    Unzips a zip file to the specified directory.

    :param zip_path: Path to the zip file.
    :param extract_to: Directory to extract the files to.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__=="__main__":
    # Example usage:
    unzip_file('data/ieee-phm-2012-data-challenge-dataset-master.zip', 'data/ieee-RUL')

    # import pandas as pd
    # df = pd.read_csv('data/ieee-RUL/Learning_set/Bearing1_1/acc_00003.csv')
    # print(df.shape)