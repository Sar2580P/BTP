import glob
import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from utils import read_yaml
from tqdm import tqdm
import pickle
import numpy as np
 
class RepresentationLearningDataset(Dataset):

    def __init__(self, df_path, sensor_name, transform=None):
        self.df = pd.read_csv(df_path)  
        self.sensor_name = sensor_name
        self.transform = transform
       
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        
        # Get the row for the sample
        file_path = self.df.loc[idx, 'file_path']
        
        # load pickle file (.pkl)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        time = 2 * (data['time'] - 0) / (20000 - 0) - 1  # scaled to [-1, 1]
 
        
        if self.sensor_name=='acc_horizontal':
            fft = data['acc_h_fft']
            time_features = data['acc_h_time_features']
            freq_features = data['acc_h_freq_features']
        elif self.sensor_name=='acc_vertical':
            fft = data['acc_v_fft']
            time_features = data['acc_v_time_features']
            freq_features = data['acc_v_freq_features']
            
        else:
            raise ValueError("Invalid sensor name. Choose either 'acc_horizontal' or 'acc_vertical'.")
        features = np.concatenate((fft, time_features, freq_features), axis=0)[:-2]
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        time = torch.tensor(time, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        
        # Apply transformation if provided
        if self.transform:
            features = self.transform(features)
        
        return features, time           
    
    
def get_dataloaders(data_config):
    tr_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(data_config['tr_path'], data_config['sensor_name']),
        batch_size=data_config['BATCH_SIZE'], shuffle=True, num_workers=data_config['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(data_config['val_path'], data_config['sensor_name']),
        batch_size=data_config['BATCH_SIZE'], shuffle=False, num_workers=data_config['num_workers']
    )
    tst_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(data_config['tst_path'], data_config['sensor_name']),
        batch_size=data_config['BATCH_SIZE'], shuffle=False, num_workers=data_config['num_workers']
    )
    return tr_loader, val_loader, tst_loader


def data_split(train_bearings, val_bearings, test_bearings):
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # get full paths of vibration_FE for each bearing
    train_vibration_FE = []
    for bearing in tqdm(train_bearings, desc="Loading train bearings..."):
        train_vibration_FE += glob.glob(f"data/ieee-RUL/*/{bearing}_vibration_FE/*")
        
    val_vibration_FE = []
    for bearing in tqdm(val_bearings, desc = "Loading val bearings..."):
        val_vibration_FE += glob.glob(f"data/ieee-RUL/*/{bearing}_vibration_FE/*")
        
    test_vibration_FE = []
    for bearing in tqdm(test_bearings, desc = "Loading test bearings..."):
        test_vibration_FE += glob.glob(f"data/ieee-RUL/*/{bearing}_vibration_FE/*")
    
    # create dataframes
    train_df['file_path'] = train_vibration_FE
    val_df['file_path'] = val_vibration_FE
    test_df['file_path'] = test_vibration_FE
    
    # add time labels
    train_df['time'] = train_df['file_path'].apply(lambda x: 10*int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    val_df['time'] = val_df['file_path'].apply(lambda x: 10*int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    test_df['time'] = test_df['file_path'].apply(lambda x: 10*int(x.split('/')[-1].split('_')[-1].split('.')[0]))
    
    # add bearing labels
    train_df['bearing'] = train_df['file_path'].apply(lambda x: x.split('/')[-2])
    val_df['bearing'] = val_df['file_path'].apply(lambda x: x.split('/')[-2])
    test_df['bearing'] = test_df['file_path'].apply(lambda x: x.split('/')[-2])
    
    # save dataframes
    train_df.to_csv('data/ieee-RUL/train_df.csv', index=False)
    val_df.to_csv('data/ieee-RUL/val_df.csv', index=False)
    test_df.to_csv('data/ieee-RUL/test_df.csv', index=False)
    return train_df, val_df, test_df


if __name__=="__main__":
    
    # check if train_df.csv, val_df.csv and test_df.csv exist
    if not os.path.exists('data/ieee-RUL/train_df.csv'):
        config = read_yaml("modelling/representation_learning/config.yaml")['dataset_params']
        train_bearings, val_bearings, test_bearings = config['train_bearings_names'], \
                                                    config['val_bearings_names'], config['test_bearings_names']
        data_split(train_bearings, val_bearings, test_bearings)