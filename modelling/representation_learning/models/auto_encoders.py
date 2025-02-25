import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import glob
from modelling.representation_learning.models.time_embedder import TimeEmbeddingModel
from modelling.representation_learning.models.modules import get_activation, get_normalization



class SparseAutoencoder(nn.Module):
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
        super(SparseAutoencoder, self).__init__()
        assert config['hidden_dim'] > config['input_dim'], "Hidden dimension must be larger than the input dimension for sparse autoencoders."
        self.sparsity_lambda = config['sparsity_lambda']
        self.momentum = config['momentum']
        
        # Initialize the running mean activation (moving average of activations)
        self.train_running_mean_activation = nn.Parameter(torch.zeros(config['hidden_dim']), requires_grad=False)  # No gradient flow for the moving average
        self.infer_running_mean_activation = nn.Parameter(torch.zeros(config['hidden_dim']), requires_grad=False)  # No gradient flow for the moving average

        # Encoder: Fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            get_normalization(config['norm_type'], config['hidden_dim']),
            get_activation(config['activation'])
        )
        
        # Decoder: Fully connected layers (same size as input)
        self.decoder = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['input_dim'])
        )
        assert config['model_name']=="SparseAutoencoder", f"Model name must be {self.__class__.__name__}, but got {config['model_name']}"
        self.model_name = f"{self.__class__.__name__}_sensor-{config['sensor_id']}"
        self.config = config
    
    def forward(self, x:torch.Tensor, is_train_mode):
        """Forward pass through the autoencoder."""
        
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
        return decoded, sparsity_loss
    
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
        return F.kl_div(mean_activation, target_dist, reduction='mean')


class SensorFeatureFusion(nn.Module):
    """
    Convolutional Autoencoder with encoder-decoder structure
    for feature representation learning, including pooling layers.
    
    Args:
        input_dim (int): Length of the input signal.
        latent_dim (int): Latent dimension (embedding size).
        channels (int): Number of input channels (e.g., 1 for grayscale or single sensor).
        activation (str): Activation function to use ('relu', 'leaky_relu', etc.).
    """
    def __init__(self, config:dict):
        super(SensorFeatureFusion, self).__init__()

        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.channels = config['channels']
        self.config = config
        
        self.base_autoencoders = self.get_base_autoencoders(config['sensor_id_list'])
        self.model_name = f"{self.__class__.__name__}"

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            get_activation(config['activation']),
            
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            get_activation(config['activation']),
            
            nn.MaxPool1d(kernel_size=2, stride=2), 

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            get_activation(config['activation']),

            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(128, self.latent_dim, kernel_size=3, stride=1, padding=1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.latent_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            get_activation(config['activation']),

            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            get_activation(config['activation']),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            get_activation(config['activation']),

            nn.ConvTranspose1d(32, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    def get_base_autoencoders(self, sensor_id_list:list):
        # TODO : debug this function
        base_autoencoders = nn.ModuleList()
        BASE_DIR = "results/autoencoders/"
        for sensor_id in sensor_id_list:
            ckpt_file_path = glob.glob(os.path.join(BASE_DIR, f"SparseAutoencoder_sensor-{sensor_id}*.pt"))[0]
            ckpt = torch.load(ckpt_file_path)
            sparse_model_config = ckpt['state_dict']['config']
            model = SparseAutoencoder(sparse_model_config, sensor_id)  # getting model config from ckpt stored....
            model.load_state_dict(ckpt['state_dict']['model'])
            base_autoencoders.append(model.encoder)
            
            del ckpt
            del model
        return base_autoencoders
    
    def forward(self, x):
        """Forward pass through the autoencoder."""
        B, C, L = x.size()
        y = x.chunk(C, dim=1).squeeze(1)
        encoded = []
        for i in range(C):
            encoded.append(self.base_autoencoders[i](y[i]))
        encoded = torch.cat(encoded, dim=1)
        encoded = self.encoder(encoded)
        decoded = self.decoder(encoded)
        return decoded



if __name__=="__main__":
    # Example usage:
    input_dim = 128  # Length of the input sequence
    latent_dim = 256  # Latent dimension size
    channels = 30  # Number of input channels (e.g., 1 for single sensor data)
    config = {
        "hidden_dim": 256,
        "input_dim": input_dim,
        "sparsity_lambda": 0.05,
        "momentum" : 0.1, 
        "norm_type": "batchnorm",
        "activation": "relu",
        "model_name" : "SparseAutoencoder",
        "sensor_id" : 1
    }
    model = SparseAutoencoder(config=config)
    
    input_tensor = torch.randn(32, input_dim)
    output, sparsity_loss = model(input_tensor, is_train_mode=1)
    print(output.size(), sparsity_loss)