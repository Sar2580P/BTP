import torch
from utils import read_yaml
import wandb
from modelling.representation_learning.models.sparse_denoisingAE import DenoisingSparseAE
from modelling.representation_learning.train_utils.train_eval import RepresentationLearning
from modelling.representation_learning.dataloaders import get_dataloaders
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from modelling.callbacks import get_callbacks
import yaml
import time
from omegaconf import OmegaConf

sweep_config =read_yaml('modelling/representation_learning/train_utils/sweep_config.yaml')
sweep_config = OmegaConf.to_container(sweep_config, resolve=True)
sweep_id = wandb.sweep(sweep_config, project="SensorRepresentationLearning")

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        wandb_config = wandb.config

        original_config_filepath = "modelling/representation_learning/train_utils/sweep_full_config.yaml"

        # ✅ First, read the existing YAML config
        original_config = yaml.safe_load(open(original_config_filepath, 'r'))

        # ✅ Then, update and write it back
        original_config['training_params'].update(wandb_config)
        # print(original_config)

        with open(original_config_filepath, "w") as f:
            yaml.dump(original_config, f)

        time.sleep(0.05)
        config = read_yaml(original_config_filepath)

        train_config = config['training_params']
        model_config = config[train_config['model_name']+'_params']
        data_config = config['dataset_params']


        # loading the model
        if train_config['model_name'] == "DenoisingSparseAE":
            model = DenoisingSparseAE(config['DenoisingSparseAE_params'])
        else:
            raise ValueError(f"model_name: {train_config['model_name']} not supported")
        model_obj = RepresentationLearning(model,config)

        tr_loader, val_loader , tst_loader = get_dataloaders(tr_df_path=data_config['tr_path'],
                                                     val_df_path=data_config['val_path'] ,
                                                     tst_df_path=data_config['tst_path'],
                                                     data_config=data_config)

        NAME = model.model_name+"_sweep"
        denoisingAE_lr, sparseAE_lr = model_config['denoisingAE_params']['lr'], model_config['sparseAE_params']['lr']
        run_name = f"denoisingAELr--{denoisingAE_lr}__sparseAELr--{sparseAE_lr}__bs--{train_config['BATCH_SIZE']}__decay-{train_config['weight_decay']}"
        wandb_logger = WandbLogger(project=NAME, name=run_name)

        early_stop_callback, _, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callback_params'])
        torch.set_float32_matmul_precision('high')
        trainer = Trainer(callbacks=[early_stop_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                          accelerator='gpu', max_epochs=train_config['MAX_EPOCHS'], logger=[wandb_logger] ,
                          accumulate_grad_batches=train_config['GRAD_ACCUMULATION_STEPS'])

        trainer.fit(model_obj, tr_loader, val_loader)
        trainer.test(model_obj, tst_loader)

# Run the sweep
wandb.agent(sweep_id, function=train, count = 50)