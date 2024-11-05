from modelling.dataloaders import (stage_classifier_tr_loader as tr_loader, 
                                   stage_classifier_val_loader as val_loader, 
                                   stage_classifier_test_loader as test_loader)
from modelling.models import Stage_Classifier
from modelling.stage_classifier.train_eval import StackedDenoisingAutoEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import os
from modelling.callbacks import (early_stop_callback, checkpoint_callback,
                                rich_progress_bar, rich_model_summary)
from utils import read_yaml
import torch
import pickle 

torch.set_float32_matmul_precision('high')  # 'medium' | 'high'
config = read_yaml('modelling/stage_classifier/config.yaml')
model = Stage_Classifier(config=config)
training_setup = StackedDenoisingAutoEncoder(config=config, model=model)
#_____________________________________________________________________________________________________________
pattern = f"{model.name}_numClasses-{config['Stage_classifier']['num_classes']}"

checkpoint_callback.dirpath = os.path.join(config['save_dir'], 'ckpts')
checkpoint_callback.filename = pattern + f"_{config['ckpt_file_name']}"

run_name = f"lr-{config['lr']}__bs-{config['BATCH_SIZE']}__decay-{config['weight_decay']}"
wandb_logger = WandbLogger(project= f"{model.name}", name=config['Stage_classifier']['num_classes'])
csv_logger = CSVLogger(config['save_dir']+'/logs/'+  pattern)

#_____________________________________________________________________________________________________________
torch.cuda.empty_cache()
trainer = Trainer(callbacks=[early_stop_callback, checkpoint_callback, rich_progress_bar, rich_model_summary],
                accelerator = 'gpu' ,max_epochs=config['MAX_EPOCHS'], logger=[wandb_logger, csv_logger] ,
                accumulate_grad_batches=config['GRAD_ACCUMULATION_STEPS'])

trainer.fit(model=training_setup  , train_dataloaders=tr_loader,
            val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader)

with open(f"{config['save_dir']}/evaluations/{pattern}.pkl", 'w') as f:
    pickle.dump(training_setup.y_hat, f)
    pickle.dump(training_setup.y_true, f)
        