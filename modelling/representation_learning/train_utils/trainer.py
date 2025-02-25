from modelling.representation_learning.models.sparse_denoisingAE import DenoisingSparseAE
from modelling.representation_learning.train_utils.train_eval import RepresentationLearning
from modelling.representation_learning.dataloaders import get_dataloaders
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from modelling.callbacks import get_callbacks
from utils import read_yaml
import torch

torch.set_float32_matmul_precision('high')  # 'medium' | 'high'
model_name = "SparseAutoencoder"
config = read_yaml('modelling/representation_learning/config.yaml')
train_config = config['training_params']
model_config = config[train_config['model_name']+'_params']
data_config = config['dataset_params']


# loading the model
if train_config['model_name'] == "DenoisingSparseAE":
    model = DenoisingSparseAE(config['DenoisingSparseAE_params'])
else:
    raise ValueError(f"model_name: {train_config['model_name']} not supported")
model_obj = RepresentationLearning(model,config)

# setting up the dataloaders
tr_loader, val_loader , tst_loader = get_dataloaders(tr_df_path=data_config['tr_path'],  
                                                     val_df_path=data_config['tr_path'] , 
                                                     tst_df_path=data_config['tr_path'],
                                                     data_config=data_config)

#_____________________________________________________________________________________________________________
early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor = get_callbacks(config['callback_params'])
checkpoint_callback.dirpath = os.path.join(train_config['save_dir'], 'ckpts')
checkpoint_callback.filename = f"{model.model_name}_"+train_config['ckpt_file_name']

wandb_logger = WandbLogger(project= "SensorRepresentationLearning", name = model.model_name)
csv_logger = CSVLogger(train_config['save_dir']+'/logs/'+ model.model_name)

#_____________________________________________________________________________________________________________
torch.cuda.empty_cache()
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary, lr_monitor],
                accelerator = 'cpu' ,max_epochs=train_config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger] ,
                accumulate_grad_batches=train_config['GRAD_ACCUMULATION_STEPS'])

trainer.fit(model=model_obj  , train_dataloaders=tr_loader,
            val_dataloaders=val_loader)
trainer.test(model=model_obj, dataloaders=tst_loader)

