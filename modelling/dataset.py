import glob
import pandas as pd 
import torch.utils.data as DataSet
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
    
class SDA_Dataset(DataSet):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return torch.tensor(np.load(self.df.iloc[idx]["full_fft_path"]))
    
    
if __name__ == "__main__":
    if not os.path.exists("data/ieee-RUL/Full_Test_Set.csv"):
        get_data_mapping()