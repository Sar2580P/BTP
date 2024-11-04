
from torch.utils.data import DataLoader
from modelling.dataset import SDA_Dataset
import pandas as pd
from utils import read_yaml

config = read_yaml("modelling/config.yaml")
BASE_DIR = "data/ieee-RUL"

sda_config = config['stage_1']['SDA_params']
sda_tr_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Learning_set.csv"))
sda_val_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Test_set.csv"))  
sda_test_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Full_Test_Set.csv"))

sda_tr_loader = DataLoader(sda_tr_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=True, num_workers=4)
sda_val_loader = DataLoader(sda_val_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=False, num_workers=4)
sda_test_loader = DataLoader(sda_test_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=False, num_workers=4)
# __________________________________________________________________________________________________________________________