import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

loss_fn = {
    "MSELoss": nn.MSELoss(),
    "HuberLoss" : nn.HuberLoss(reduction='mean', delta=1.0)
}

class RepresentationLearning(pl.LightningModule):
    def __init__(self, model, config):
        super(RepresentationLearning, self).__init__()
        self.model:nn.Module = model
        self.config = config
        self.model_name = config['training_params']['model_name']
        self.MAX_EPOCHS = config['training_params']['MAX_EPOCHS']
        self.loss_fn_name = config['training_params']['loss_fn']
        self.criterion = loss_fn[self.loss_fn_name]

        self.save_hyperparameters(ignore=['model'])  # Saves the hyperparameters for logging


    def training_step(self, batch, batch_idx):
        if "SparseAutoencoder" in self.model_name:
            decoded_output , kl_loss = self.model(batch, is_train_mode = 1)
            self.log("train_kl_loss", kl_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            loss = self.criterion(decoded_output, batch) + self.kl_scheduler(self.current_epoch)*kl_loss
        elif "SensorFeatureFusion" in self.model_name:
            decoded_output = self.model(batch)
            loss = self.criterion(decoded_output, batch)
        elif "DenoisingSparseAE" in self.model_name:
            x, t = batch
            decoded_output, sparsity_loss = self.model.forward(x, t , self.current_epoch, self.MAX_EPOCHS , is_train_mode = 1)
            loss = self.criterion(decoded_output, x) + (sparsity_loss*self.kl_scheduler(self.current_epoch, self.MAX_EPOCHS) if sparsity_loss is not None else 0)
            if sparsity_loss is not None :
                self.log("train_sparsity_loss", sparsity_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        self.log(f"train_{self.loss_fn_name}", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if "SparseAutoencoder" in self.model_name:
            decoded_output , kl_loss = self.model(batch, is_train_mode = 0)
            self.log("val_kl_loss", kl_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            loss = self.criterion(decoded_output, batch) + self.kl_scheduler(self.current_epoch)*kl_loss
        elif "SensorFeatureFusion" in self.model_name:
            decoded_output = self.model(batch)
            loss = self.criterion(decoded_output, batch)

        elif "DenoisingSparseAE" in self.model_name:
            x, t = batch
            decoded_output, sparsity_loss = self.model.forward(x, t , self.current_epoch, self.MAX_EPOCHS , is_train_mode = 0)
            loss = self.criterion(decoded_output, x) + (sparsity_loss*self.kl_scheduler(self.current_epoch, self.MAX_EPOCHS) if sparsity_loss is not None else 0)
            if sparsity_loss is not None :
                self.log("val_sparsity_loss", sparsity_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        self.log(f"val_{self.loss_fn_name}", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if "SparseAutoencoder" in self.model_name:
            decoded_output , kl_loss = self.model(batch, is_train_mode = 0)
            self.log("test_kl_loss", kl_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
            loss = self.criterion(decoded_output, batch) + self.kl_scheduler(self.current_epoch)*kl_loss
        elif "SensorFeatureFusion" in self.model_name:
            decoded_output = self.model(batch)
            loss = self.criterion(decoded_output, batch)
        elif "DenoisingSparseAE" in self.model_name:
            x, t = batch
            decoded_output, sparsity_loss = self.model.forward(x, t , self.current_epoch, self.MAX_EPOCHS , is_train_mode = 0)
            loss = self.criterion(decoded_output, x) + (sparsity_loss*self.kl_scheduler(self.current_epoch, self.MAX_EPOCHS) \
                                                        if sparsity_loss is not None else 0)
            if sparsity_loss is not None :
                self.log("tst_sparsity_loss", sparsity_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)

        self.log(f"tst_{self.loss_fn_name}", loss, on_epoch=True, on_step=False,prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.layer_lr, lr=1e-5,
                                 weight_decay = self.config['training_params']['weight_decay'])
        scheduler_name = self.config['lr_scheduler_params']['scheduler_name']
        scheduler_params = self.config['lr_scheduler_params'][f"{scheduler_name}_params"]
        if scheduler_name == 'exponential_decay_lr_scheduler':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, **scheduler_params)

        elif scheduler_name == 'cosine_annealing_lr_scheduler':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, **scheduler_params)

        else:
            raise ValueError(f"Scheduler: {scheduler_name} not supported, available schedulers are: ['exponential_decay_lr_scheduler', 'cosine_annealing_lr_scheduler']")

        return [optim], [{'scheduler': lr_scheduler, 'interval': 'step',
                          'monitor': f"train_{self.loss_fn_name}" ,
                          'name': scheduler_name}]



    def kl_scheduler(self, epoch, total_epochs, start=0.005, end=0.05):
        """Smoothly increases KL coefficient from start to end over epochs."""
        # Control the growth rate of the curve (higher means faster saturation)
        growth_rate = 25

        # Sigmoid-shaped curve to control the transition
        progress = epoch / total_epochs
        kl_weight = start + (end - start) * (1 / (1 + math.exp(-growth_rate * (progress - 0.5))))

        return kl_weight

