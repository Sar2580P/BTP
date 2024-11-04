import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn

loss_fn = {
    "SDA_squared_error": nn.MSELoss(),
}

class StackedDenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, model, config):
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.model:nn.Module = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])  # Saves the hyperparameters for logging
        self.criterion = loss_fn[self.config['loss_fn']]

    def training_step(self, batch, batch_idx):
        y, y_hat = self.model.forward(batch)
        loss = self.criterion(y_hat, y)
        self.log(f"train_{self.config['loss_fn']}", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, y_hat = self.model.forward(batch)
        loss = self.criterion(y_hat, y)
        self.log(f"val_{self.config['loss_fn']}", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        y, y_hat = self.model.forward(batch)
        loss = self.criterion(y_hat, y)
        self.log(f"test_{self.config['loss_fn']}", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.config['lr'])
        return optimizer
