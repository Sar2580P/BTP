import glob
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
import os

def get_data_mapping():
    BASE_DIR = "data/ieee-RUL"
    for dataset in ["Full_Test_Set", "Learning_set", "Test_set"]:
        df = pd.DataFrame(columns = ["full_fft_path"])
        bearing_fft_folders = glob.glob(f"{BASE_DIR}/{dataset}/Bearing*_fft")
        print(f"Found {len(bearing_fft_folders)} files in {dataset}")
        for bearing_fft_folder in bearing_fft_folders:
            files = glob.glob(f"{bearing_fft_folder}/*.npy")
            for file in files:
                df = pd.concat([df, pd.DataFrame({"full_fft_path":[file]})])
        df.to_csv(f"{BASE_DIR}/{dataset}.csv", index=False)

    print("Data mapping created successfully")

class SDA_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        x =  torch.tensor(np.load(self.df.iloc[idx]["full_fft_path"]), dtype=torch.float32)
        return x

class Stage_Classifier_Dataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        n = self.config['Stage_classifier']['num_classes']
        assert 1<n and n<8 , f"num_classes should lie [2,7] , but got {self.config['Stage_classifier']['num_classes']}"
        self.Y = self.df[f"stage_{self.config['Stage_classifier']['num_classes']}"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        full_path = os.path.join(self.df.iloc[idx]['dir'], self.df.iloc[idx]["file_name"]+'.npy')
        x =  torch.tensor(np.load(full_path), dtype=torch.float32)
        y = torch.tensor(self.Y.iloc[idx])
        return x, y


if __name__ == "__main__":
    if not os.path.exists("data/ieee-RUL/Full_Test_Set.csv"):
        get_data_mapping()