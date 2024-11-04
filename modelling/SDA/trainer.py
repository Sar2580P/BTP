from modelling.dataloaders import sda_tr_loader, sda_val_loader, sda_test_loader
from modelling.models import SDA_Initialiser
from modelling.train_eval import LightningModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from modelling.callbacks import (early_stop_callback, checkpoint_callback,
                                rich_progress_bar, rich_model_summary)
from utils import read_yaml
import torch

torch.set_float32_matmul_precision('medium')  #  | 'high'
config = read_yaml('modelling/SDA/config.yaml')
model = SDA_Initialiser(config=config['SDA_params'])
training_setup = LightningModel(config=config['SDA_params'], model=model)
#_____________________________________________________________________________________________________________

checkpoint_callback.dirpath = os.path.join(config['save_dir'], 'ckpts', model.name)
checkpoint_callback.filename = config['ckpt_file_name']

run_name = f"lr-{config['lr']}__bs-{config['BATCH_SIZE']}__decay-{config['weight_decay']}"
wandb_logger = WandbLogger(project= f"{model.name}", name = config['ckpt_file_name'])
csv_logger = CSVLogger(config['save_dir']+f"/{model.name}/"+'/logs/'+  config['ckpt_file_name'])

#_____________________________________________________________________________________________________________
torch.cuda.empty_cache()
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                  accelerator = 'gpu' ,max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger] ,
                  accumulate_grad_batches=config['GRAD_ACCUMULATION_STEPS'])

trainer.fit(model=training_setup  , train_dataloaders=sda_tr_loader,
            val_dataloaders=sda_val_loader , ckpt_path='last')
trainer.test(dataloaders=sda_test_loader , ckpt_path='last')