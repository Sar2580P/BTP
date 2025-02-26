import glob
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch

class RepresentationLearningDataset(Dataset):
    """"
    Args:
        df_path (str): Path to the CSV file containing the file paths.
        sensor_idx (int): Index of the sensor (i.e. column index) to load.
        transform (callable, optional): Optional transform to be applied on the FFT data.
    """
    def __init__(self, df_path, sensor_idx, bearings_list, transform=None):
        self.df = pd.read_csv(df_path)
        self.sensor_idx = sensor_idx
        self.transform = transform

        # filter rows of df to contain rows which has bearing in bearings_list
        self.df = self.df[self.df['full_fft_path'].apply(lambda x: x.split('/')[-2][:-4] in bearings_list)].reset_index(drop=True)
        print(len(self.df))
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        For the given index, load the .npy file corresponding to the specified sensor index.

        Returns:
            torch.Tensor: The FFT data (as a torch float tensor) from the selected sensor.
        """
        # Get the row for the sample
        file_path = self.df.iloc[idx, 0]
        fft_data = np.load(file_path)[self.sensor_idx]

        # Apply an optional transform if provided
        if self.transform:
            fft_data = self.transform(fft_data)

        # Convert the numpy array to a torch tensor (ensuring it is of type float)
        fft_tensor = torch.tensor(fft_data, dtype=torch.float)
        time_stamp = torch.tensor((int)(file_path.split('/')[-1].split('_')[-1].split('.')[0]))
        return torch.unsqueeze(fft_tensor, dim=0), torch.unsqueeze(time_stamp, dim=0)


def get_dataloaders(tr_df_path,  val_df_path , tst_df_path, data_config):
    tr_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(tr_df_path, data_config['sensor_idx'], data_config['train_bearings_names']),
        batch_size=data_config['BATCH_SIZE'], shuffle=True, num_workers=data_config['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(val_df_path, data_config['sensor_idx'], data_config['val_bearings_names']),
        batch_size=data_config['BATCH_SIZE'], shuffle=False, num_workers=data_config['num_workers']
    )
    tst_loader = torch.utils.data.DataLoader(
        RepresentationLearningDataset(tst_df_path, data_config['sensor_idx'], data_config['test_bearings_names']),
        batch_size=data_config['BATCH_SIZE'], shuffle=False, num_workers=data_config['num_workers']
    )
    return tr_loader, val_loader, tst_loader