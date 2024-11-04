from modelling.dataloaders import sda_tr_loader, sda_val_loader, sda_test_loader
from modelling.models import SDA_Initialiser
from modelling.SDA.train_eval import StackedDenoisingAutoEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from modelling.callbacks import (early_stop_callback, checkpoint_callback,
                                rich_progress_bar, rich_model_summary)
from utils import read_yaml
import torch

torch.set_float32_matmul_precision('high')  # 'medium' | 'high'
config = read_yaml('modelling/SDA/config.yaml')
model = SDA_Initialiser(config=config)
training_setup = StackedDenoisingAutoEncoder(config=config, model=model)
#_____________________________________________________________________________________________________________
if model.ckpt_file_name is not None:

    checkpoint_callback.dirpath = os.path.join(config['save_dir'], 'ckpts')
    checkpoint_callback.filename = model.ckpt_file_name+ f"_{config['ckpt_file_name']}"

    run_name = f"lr-{config['lr']}__bs-{config['BATCH_SIZE']}__decay-{config['weight_decay']}"
    wandb_logger = WandbLogger(project= f"{model.name}", name = model.ckpt_file_name)
    csv_logger = CSVLogger(config['save_dir']+'/logs/'+  model.ckpt_file_name)

    #_____________________________________________________________________________________________________________
    torch.cuda.empty_cache()
    trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                    accelerator = 'gpu' ,max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger] ,
                    accumulate_grad_batches=config['GRAD_ACCUMULATION_STEPS'])

    trainer.fit(model=training_setup  , train_dataloaders=sda_tr_loader,
                val_dataloaders=sda_val_loader)
    trainer.test(dataloaders=sda_test_loader)

else :
    print("All layers initialized using Stacked Denoising Autoencoder")