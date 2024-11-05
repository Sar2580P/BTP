import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn as nn
import torchmetrics
import glob
import pickle

class StackedDenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, model, config):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.model:nn.Module = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])  # Saves the hyperparameters for logging
        self.tr_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'], weights = 'quadratic')
        self.val_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'], weights = 'quadratic')
        self.tst_kappa = torchmetrics.CohenKappa(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'], weights = 'quadratic')

        self.tr_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'])
        self.val_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'])
        self.tst_accuracy = torchmetrics.Accuracy(task = 'multiclass' , num_classes = self.config['Stage_classifier']['num_classes'])
        self.criterion = nn.CrossEntropyLoss()

        self.y_hat, self.y_true = [], []

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y.long())
        self.log(f"train_CE_loss", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"train_kappa", self.tr_kappa(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"train_accuracy", self.tr_accuracy(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y.long())
        self.log(f"val_CE_loss", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"val_kappa", self.val_kappa(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"val_accuracy", self.val_accuracy(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        self.y_hat.append(y_hat)
        self.y_true.append(y.long())

        loss = self.criterion(y_hat, y)
        self.log(f"test_CE_loss", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"test_kappa", self.tst_kappa(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        self.log(f"test_accuracy", self.tst_accuracy(y_hat, y), on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer

def find_optimal_stage_classes():
    BASE_DIR = "results/evaluations"
    pattern = "Stage_classifier_numClasses-"
    files = glob.glob(f"{BASE_DIR}/{pattern}*.pkl")

    for file in files:
        with open(file, 'r') as f:
            y_hat = pickle.load(f)
            y_true = pickle.load(f)
            print(f"{file} : {y_hat} , {y_true}")