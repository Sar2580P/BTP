import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn

loss_fn = {
    "SDA_squared_error": nn.MSELoss(),
}

class LightningModel(pl.LightningModule):
    def __init__(self, model, config):
        super(LightningModel, self).__init__()
        self.model:nn.Module = model
        self.config = config
        self.save_hyperparameters()  # Saves the hyperparameters for logging
        self.criterion = loss_fn[self.config['loss_fn']]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y)
        self.log(self.config['loss_fn'], loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y)
        self.log(self.config['loss_fn'], loss, on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.forward(x)
        loss = self.criterion(y_hat, y)
        self.log(self.config['loss_fn'], loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        return optimizer
