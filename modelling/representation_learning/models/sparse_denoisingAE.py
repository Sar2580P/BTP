import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, List
from utils import read_yaml
from modelling.representation_learning.models.modules import (DenoisingEncoder,
                                                        DenoisingDecoder, 
                                                        SparseEncoder,
                                                        SparseDecoder, 
                                                        plot_model
                                                        )
from modelling.representation_learning.models.time_embedder import TimeEmbedding

class DenoisingAE(nn.Module):
    def __init__(self, config):
        super(DenoisingAE, self).__init__()
        
        self.encoder = DenoisingEncoder(**config['encoder_params'])
        self.decoder = DenoisingDecoder(**config['decoder_params'])
        if config['apply_timeEmbedding']: 
            self.time_embedder = TimeEmbedding(**config['time_embedder_params'])
        else : 
            self.time_embedder = None
        self.model_name = f"{self.__class__.__name__}_sensor-{config['sensor_name']}"
        
    def forward(self, x:torch.Tensor, time_stamp:torch.Tensor):
        latent, encoder_outputs = self.forward_encoder(x, time_stamp)
        reconstruction = self.forward_decoder(latent, encoder_outputs)
        return reconstruction
    
    def forward_encoder(self, x:torch.Tensor, time_stamp:torch.Tensor):
        if self.time_embedder is None:
            return self.encoder(x)
        time_embedding = self.time_embedder(time_stamp).unsqueeze(1)
        print(f"Time embedding shape: {time_embedding.shape}, x shape: {x.shape} ----------<<<<<<<>>>>>>>>>>>")
        return self.encoder(torch.add(x, time_embedding))
    
    def forward_decoder(self, x:torch.Tensor, encoder_outputs:List[torch.Tensor]):
        return self.decoder(x, encoder_outputs=encoder_outputs)
    
    
class SparseAE(nn.Module):
    """
    Sparse Autoencoder with one hidden layer where sparsity is enforced.
    
    Args:
        input_dim (int): Dimension of the input features (len, the length of the signal).
        hidden_dim (int): Dimension of the hidden layer, which should be larger than input_dim.
        activation (str): Activation function to use.
        norm_type (str): Normalization method ("batchnorm", "groupnorm", "instancenorm", or "none").
        sparsity_lambda (float): Regularization strength for sparsity.
        beta (float): Weight for KL divergence loss.
        momentum (float): Momentum for the weighted moving average of hidden activations.
    """
    def __init__(self, config:dict):
        super(SparseAE, self).__init__()
        assert config['sparse_hidden_dim'] > config['input_dim'], "sparse_hidden_dim must be larger than the input dimension for sparse autoencoders."
        self.sparsity_lambda = config['sparsity_lambda']
        self.momentum = config['momentum']
        
        # Initialize the running mean activation (moving average of activations)
        self.train_running_mean_activation = nn.Parameter(torch.zeros(config['sparse_hidden_dim']), requires_grad=False)  # No gradient flow for the moving average
        self.infer_running_mean_activation = nn.Parameter(torch.zeros(config['sparse_hidden_dim']), requires_grad=False)  # No gradient flow for the moving average

        
        # Build sparse branch (fully connected layers).
        self.encoder = SparseEncoder(
            input_dim=config['input_dim'],
            hidden_dim=config['sparse_hidden_dim'],
            num_layers=1
        )
        self.decoder = SparseDecoder(
            input_dim=config['sparse_hidden_dim'],
            output_dim=config['out_dim'],
            num_layers=1
        )
        
        self.model_name = f"{self.__class__.__name__}_sensor-{config['sensor_name']}"
        self.config = config
    
    def forward(self, x:torch.Tensor, is_train_mode):
        """Forward pass through the autoencoder."""
        B, C, L = x.size()
        # convert to shape: (B, C, L) --> (B*c, L)
        x = x.view(B*C, L)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        activation_mean = encoded.mean(dim=0)  # Mean of activations for each hidden unit       
        if is_train_mode==1: # Training mode
            self.train_running_mean_activation.data = self.momentum * self.train_running_mean_activation.data + \
                                                (1 - self.momentum) * activation_mean
            running_mean_activation = self.train_running_mean_activation
        else:
            self.infer_running_mean_activation.data = self.momentum * self.infer_running_mean_activation.data + \
                                                (1 - self.momentum) * activation_mean
            running_mean_activation = self.infer_running_mean_activation
        
        # (Sparsity enforcement)
        sparsity_loss = self.kl_divergence_loss(running_mean_activation, self.sparsity_lambda)   
        
        # convert back to shape: (B*c, L) --> (B, C, L)
        decoded = decoded.view(B, C, L)
        return decoded, sparsity_loss
    
    def forward_encoder(self, x:torch.Tensor):
        return self.encoder(x)
        
    def kl_divergence_loss(self, mean_activation, target_sparsity=0.05):
        """
        Calculate the KL divergence loss for sparsity enforcement using built-in KL divergence.
        Enforces the hidden activations to be sparse (close to zero on average).
        """
        # Convert the target sparsity into a Bernoulli distribution with probability target_sparsity
        target_dist = torch.full_like(mean_activation, target_sparsity)
        
        # KL Divergence: We need to calculate the KL divergence between the activation distribution
        # and a target sparsity distribution.
        # First, make sure to apply log to the activations since KLDiv expects log probabilities
        return F.kl_div(mean_activation, target_dist, reduction='sum')
    
    
#############################################
# Orchestrator: SparseDenoisingAE
#############################################
class DenoisingSparseAE(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Sparse Denoising Autoencoder that orchestrates both denoising and sparse branches.
        
        The autoencoder can operate in two modes:
          - "denoise": Standard denoising autoencoder.
          - "sparse": Input is encoded via the denoising encoder, then passed through a sparse
                      branch (with sparsity enforced via a KL divergence penalty) before being decoded.
        """
        super().__init__()
        self.train_sparseAE_flag = False
        self.is_denoisingAE_frozen = False

        self.denoisingAE = DenoisingAE(config['denoisingAE_params'])
        self.sparseAE = SparseAE(config['sparseAE_params'])
        
        self.layer_lr = [{'params' : self.denoisingAE.parameters(), 'lr' : config['denoisingAE_params']['lr']},
                         {'params': self.sparseAE.parameters(), 'lr': config['sparseAE_params']['lr']}]
        
        self.model_name: str = f"{self.__class__.__name__}_sensor-{config['sensor_name']}"
        self.config = config

    def forward_denoise(self, x: torch.Tensor, time_stamp: torch.Tensor) -> Tuple[torch.Tensor, 
                                                                                  Optional[torch.Tensor]]:
        """
        Forward pass using only the denoising autoencoder.
        
        Args:
            x (torch.Tensor): Input signal of shape (batch, in_channels, seq_len).
            time_stamp (torch.Tensor): Time stamp of shape (batch, 1).
            is_train_mode (int): 1 for training, 0 for inference.
            
        Returns:
            Tuple containing the reconstruction and None (no sparsity loss).
        """
        reconstruction = self.denoisingAE.forward(x, time_stamp)
        return reconstruction

    def forward_sparse(self, x: torch.Tensor, is_train_mode:int=0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sparseAE.forward(x, is_train_mode)
    
    

    def forward(self, x: torch.Tensor, time_stamp: torch.Tensor = None, 
                curr_epoch=0 , total_epoch=10, is_train_mode=0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Main forward method.
        
        Args:
            x (torch.Tensor): Input sensory signal of shape (batch, in_channels, seq_len).
            time_stamp (torch.Tensor): Time stamp tensor of shape (batch, 1).
            mode (str): Either "denoise" or "sparse".
            is_train_mode (int): 1 for training mode, 0 for inference.
            
        Returns:
            Tuple containing:
              - Reconstructed signal (batch, in_channels, seq_len)
              - Sparsity loss if in "sparse" mode; otherwise None.
        """
        self.train_sparseAE_flag = self.should_train_sparseAE(curr_epoch, total_epoch)
        if is_train_mode==1:
            if self.train_sparseAE_flag:
                # freeze denoisingAE
                if not self.is_denoisingAE_frozen:
                    self.is_denoisingAE_frozen = True
                    for param in self.denoisingAE.parameters():
                        param.requires_grad = False
                denoised_latent, encoder_outputs = self.denoisingAE.forward_encoder(x, time_stamp)
                sparseAE_reconstruction, sparsity_loss = self.forward_sparse(denoised_latent, is_train_mode=is_train_mode)
                denoisingAE_reconstruction = self.denoisingAE.forward_decoder(sparseAE_reconstruction, encoder_outputs)
                return denoisingAE_reconstruction, sparsity_loss
            
            else : 
                if self.is_denoisingAE_frozen:
                    self.is_denoisingAE_frozen = False
                    for param in self.denoisingAE.parameters():
                        param.requires_grad = True
                return self.forward_denoise(x, time_stamp), None
            
        else: 
            if not self.train_sparseAE_flag:
                return self.forward_denoise(x, time_stamp), None
            else:
                denoised_latent , encoded_outputs = self.denoisingAE.forward_encoder(x, time_stamp)
                sparseAE_reconstruction, sparsity_loss = self.forward_sparse(denoised_latent, is_train_mode=is_train_mode)
                denoisingAE_reconstruction = self.denoisingAE.forward_decoder(sparseAE_reconstruction, encoder_outputs=encoded_outputs)
                return denoisingAE_reconstruction, sparsity_loss
        
    def forward_encoder(self, x: torch.Tensor, time_stamp: torch.Tensor) -> torch.Tensor:
        latent = self.denoisingAE.forward_encoder(x, time_stamp)
        sparse_latent = self.sparseAE.forward_encoder(latent)
        return sparse_latent
    
    def forward_denoising_encoder(self, x: torch.Tensor, time_stamp: torch.Tensor) -> torch.Tensor:
        return self.denoisingAE.forward_encoder(x, time_stamp)
        
    def should_train_sparseAE(self, curr_epoch, total_epoch):
        return curr_epoch > total_epoch*(2/3)  # some scheduling for training sparseAE
        # return True
            
            
if __name__=="__main__":
    print(1)
    config = read_yaml("modelling/representation_learning/config.yaml")
    print(config)
    input = torch.randn(1,1,256, device="cpu")
    time_stamp = torch.randn(1,1, device="cpu")
    print(2, "____________________________________")
    model = DenoisingSparseAE(config['DenoisingSparseAE_params']).to("cpu")
    print(3, "____________________________________")
    plot_model(config=config, model=model, input_size = (8,1,256), savefile_name="FullSparseDenoisingAE")
    y, _ = model.forward(input,time_stamp, 0, 10 ,is_train_mode=0)
    print(4, "____________________________________")
    print(y)