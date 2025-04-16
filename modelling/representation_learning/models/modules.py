import torch
from torch import nn
from typing import List, Tuple, Optional
from torchview import draw_graph

def plot_model(config, model, input_size:Tuple, savefile_name:str):
  model_graph = draw_graph(model, input_size=input_size, graph_dir ='TB', expand_nested=True,
                            graph_name=model.model_name,save_graph=True,filename=savefile_name,
                            directory='pics', depth = 3)
  model_graph.visual_graph

def get_activation(name):
    assert name.lower() in ["relu", "leaky_relu", "prelu", "elu", "gelu"], \
        "Invalid activation function. Choose from ['relu', 'leaky_relu', 'prelu', 'elu', 'gelu']"
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "prelu" : nn.PReLU(),   
        "elu": nn.ELU(),  
        "gelu": nn.GELU(),
    }
    return activations[name.lower()]

def get_normalization(norm_type, num_features, num_groups=8):
    """Returns the specified normalization layer."""
    assert norm_type.lower() in ["batchnorm", "groupnorm", "instancenorm", "none"], \
        "Invalid normalization type. Choose from ['batchnorm', 'groupnorm', 'instancenorm', 'none']"
    norm_layers = {
        "batchnorm": nn.BatchNorm1d(num_features) if num_features > 1 else nn.Identity(),
        "groupnorm": nn.GroupNorm(num_groups, num_features) if num_features > 1 else nn.Identity(),
        "instancenorm": nn.InstanceNorm1d(num_features) if num_features > 1 else nn.Identity(),
        "none": nn.Identity()
    }
    return norm_layers[norm_type.lower()]

# #############################################
# # Utility Block: Squeeze-and-Excitation
# #############################################
# class SqueezeExcitationBlock(nn.Module):
#     def __init__(self, channels: int, reduction: int = 16, activation_name:str = 'relu') -> None:
#         """
#         Squeeze-Excitation block for 1D inputs.
        
#         Args:
#             channels (int): Number of input channels.
#             reduction (int): Reduction factor.
#         """
#         super().__init__()
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.fc2 = nn.Linear(channels // reduction, channels)
#         self.activation = get_activation(activation_name)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x shape: (batch, channels, seq_len)
#         # Squeeze: global average pooling over the sequence dimension.
#         y = x.mean(dim=2)  # shape: (batch, channels)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).unsqueeze(2)  # shape: (batch, channels, 1)
#         return x * y  # Broadcast multiplication over seq_len.


#############################################
# Denoising Encoder 
#############################################
class DenoisingEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        activation_name: str = 'relu'
    ) -> None:
        """
        Convolutional encoder for denoising using max pooling for downsampling.
        
        Args:
            in_channels (int): Number of channels in the input.
            seq_len (int): Length of the input sequence.
            conv_channels (List[int]): List of output channels for each conv block.
            kernel_sizes (List[int]): List of kernel sizes.
            pool_sizes (List[int]): List of pooling kernel sizes for each block.
            latent_dim (int): Dimension of the output latent vector.
            use_se (bool): Whether to use squeeze-excitation blocks.
            activation_name (str): Name of the activation function to use.
        """
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(pool_sizes), "Conv settings must have the same length."
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        current_seq_len = seq_len
        
        # Build convolutional blocks with max pooling.
        for out_channels, kernel, pool_size in zip(conv_channels, kernel_sizes, pool_sizes):
            # Use stride=1 in convolution, then downsample with max pool.
            conv = nn.Conv1d(current_channels, out_channels, kernel_size=kernel, stride=1, padding=kernel // 2)
            bn = nn.BatchNorm1d(out_channels)
            activation = get_activation(activation_name)
            pool = nn.MaxPool1d(kernel_size=pool_size)  # Downsampling by pool_size.
            block = nn.Sequential(conv, bn, activation, pool)
            # Optionally, add squeeze-excitation here if desired.
            self.conv_layers.append(block)
            current_channels = out_channels
            # Update sequence length: max pool reduces length by floor division.
            current_seq_len = current_seq_len // pool_size
        
        # Save details for decoder.
        self._final_conv_channels = current_channels
        self._final_seq_len = current_seq_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, seq_len).
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
              - Latent representation of shape (batch, latent_dim).
              - List of feature maps from each conv block for skip connections.
        """
        skip_connections: List[torch.Tensor] = []
        latent=x
        for i,  layer in enumerate(self.conv_layers):
            latent = layer(latent)
            skip_connections.append(latent)  
            
        return latent, skip_connections

#############################################
# Denoising Decoder
#############################################
class DenoisingDecoder(nn.Module):
    def __init__(
        self,
        final_conv_channels: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        out_channels: int,
        activation_name: str = 'relu'
    ) -> None:
        """
        Convolutional decoder that mirrors the DenoisingEncoder.
        
        Args:
            latent_dim (int): Dimension of the latent representation.
            final_conv_channels (int): Number of channels at the end of the encoder.
            final_seq_len (int): Sequence length after the encoder's conv layers.
            conv_channels (List[int]): The conv_channels used in the encoder.
            kernel_sizes (List[int]): The kernel_sizes used in the encoder.
            pool_sizes (List[int]): The pool_sizes used in the encoder (for upsampling in deconv).
            out_channels (int): Number of channels in the output (should match original in_channels).
            use_se (bool): Whether to use squeeze-excitation blocks.
            activation_name (str): Name of the activation function.
        """
        super().__init__()
        
        # Build deconvolution layers (mirror of conv layers in reverse order).
        self.deconv_layers = nn.ModuleList()
        # Reverse the settings. Note: we now use pool_sizes instead of strides.
        rev_conv_channels = conv_channels[::-1]
        rev_kernel_sizes = kernel_sizes[::-1]
        rev_pool_sizes = pool_sizes[::-1]
        
        current_channels = final_conv_channels
        for i in range(len(rev_conv_channels)):
            # For the last deconv block, output the final number of channels.
            out_chan = rev_conv_channels[i] if i < len(rev_conv_channels) - 1 else out_channels
            stride_val = rev_pool_sizes[i]
            # Set output_padding=1 only if the stride is greater than 1.
            output_padding_val = 1 if stride_val > 1 else 0

            convt = nn.ConvTranspose1d(
                current_channels, 
                out_chan, 
                kernel_size=rev_kernel_sizes[i],
                stride=stride_val,  # Upsample using the pool size.
                padding=rev_kernel_sizes[i] // 2,
                output_padding=output_padding_val
            )

            bn = nn.BatchNorm1d(out_chan)
            activation = get_activation(activation_name)
            block = nn.Sequential(convt, bn, activation)

            self.deconv_layers.append(block)   
            current_channels = out_chan
            
        self.linear = nn.Linear(out_channels, out_channels)

    def forward(
        self, 
        x:torch.Tensor, 
        encoder_outputs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Latent vector of shape (batch, latent_dim).
            encoder_outputs (Optional[List[torch.Tensor]]): List of encoder feature maps for skip connections.
            
        Returns:
            torch.Tensor: Reconstructed signal of shape (batch, out_channels, seq_len).
        """
        # Reverse the encoder outputs for skip connections.
        latent = x
        n = len(encoder_outputs) if encoder_outputs is not None else 0
        for i, layer in enumerate(self.deconv_layers):
            # If encoder outputs are available, concatenate for skip connection.
            
            if i < n:  
                try:
                    latent = torch.add(latent, encoder_outputs[n-i-1])
                except: 
                    raise ValueError(f"Failed to add skip conn, decoder-{i+1}/{len(self.deconv_layers)}, latent shape: {latent.shape}, skip shape: {encoder_outputs[n-i-1].shape}")
            latent = layer(latent)
        return latent
        


#############################################
# Sparse Branch: Fully Connected Encoder/Decoder
#############################################
class SparseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        """
        Fully connected network that maps the denoising latent vector into an over–complete space.
        
        Args:
            input_dim (int): Dimension of the denoising latent representation.
            hidden_dim (int): Dimension of the sparse (over–complete) representation.
            num_layers (int): Number of FC layers.
        """
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).
            
        Returns:
            torch.Tensor: Sparse representation of shape (batch, hidden_dim).
        """
        out = self.net(x)
        # Optionally add a residual connection if dimensions match.
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        return out

class SparseDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2) -> None:
        """
        Fully connected network that maps the sparse representation back to the denoising latent space.
        
        Args:
            input_dim (int): Dimension of the sparse representation.
            output_dim (int): Target dimension (should match the denoising latent dimension).
            num_layers (int): Number of FC layers.
        """
        super().__init__()
        layers: List[nn.Module] = []
        hidden_dim = input_dim  # we keep same dimension within the sparse branch.
        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Sparse representation (batch, input_dim).
            
        Returns:
            torch.Tensor: Reconstructed latent vector (batch, output_dim).
        """
        out = self.net(x)
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        return out
