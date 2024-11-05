
from torch.utils.data import DataLoader
from modelling.dataset import SDA_Dataset, Stage_Classifier_Dataset
import pandas as pd
from utils import read_yaml

BASE_DIR = "data/ieee-RUL"

sda_config = read_yaml("modelling/SDA/config.yaml")
sda_tr_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Learning_set.csv"))
sda_val_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Test_set.csv"))
sda_test_dataset = SDA_Dataset(pd.read_csv(f"{BASE_DIR}/Full_Test_Set.csv"))

sda_tr_loader = DataLoader(sda_tr_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=True, num_workers=sda_config['num_workers'])
sda_val_loader = DataLoader(sda_val_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=False, num_workers=sda_config['num_workers'])
sda_test_loader = DataLoader(sda_test_dataset, batch_size=sda_config['BATCH_SIZE'], shuffle=False, num_workers=sda_config['num_workers'])
# __________________________________________________________________________________________________________________________

stage_classifier_config = read_yaml("modelling/stage_classifier/config.yaml")
stage_classifier_tr_dataset = Stage_Classifier_Dataset(pd.read_csv(f"{BASE_DIR}/life_cycle_staging_Test_set.csv"), stage_classifier_config)
stage_classifier_val_dataset = Stage_Classifier_Dataset(pd.read_csv(f"{BASE_DIR}/life_cycle_staging_Test_set.csv"), stage_classifier_config)
stage_classifier_test_dataset = Stage_Classifier_Dataset(pd.read_csv(f"{BASE_DIR}/life_cycle_staging_Test_set.csv"), stage_classifier_config)

stage_classifier_tr_loader = DataLoader(stage_classifier_tr_dataset, batch_size=stage_classifier_config['BATCH_SIZE'], shuffle=True, num_workers=stage_classifier_config['num_workers'])
stage_classifier_val_loader = DataLoader(stage_classifier_val_dataset, batch_size=stage_classifier_config['BATCH_SIZE'], shuffle=False, num_workers=stage_classifier_config['num_workers'])
stage_classifier_test_loader = DataLoader(stage_classifier_test_dataset, batch_size=stage_classifier_config['BATCH_SIZE'], shuffle=False, num_workers=stage_classifier_config['num_workers'])