import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim (int): Dimensionality of the time embedding.
        """
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_diff (torch.Tensor): Time difference from start in seconds. Shape: [batch_size, 1]
        Returns:
            torch.Tensor: Time embeddings. Shape: [batch_size, embedding_dim]
        """
        time_embedding = self.linear(time_diff)  # Shape [batch_size, 1] -> [batch_size, embedding_dim]
        return F.relu(time_embedding)

class MLPTimeEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Args:
            embedding_dim (int): Input dimensionality of time embedding.
            hidden_dims (list[int]): List of hidden layer sizes.
            output_dim (int): Final output dimensionality.
        """
        super(MLPTimeEncoder, self).__init__()
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_embedding (torch.Tensor): Time embeddings. Shape: [batch_size, embedding_dim]
        Returns:
            torch.Tensor: Encoded time features. Shape: [batch_size, output_dim]
        """
        return self.mlp(time_embedding)

class TimeEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Combines TimeEmbedding and MLPTimeEncoder into a single module.

        Args:
            embedding_dim (int): Time embedding dimensionality.
            hidden_dims (list[int]): Hidden layer sizes for the MLP.
            output_dim (int): Final output dimensionality.
        """
        super(TimeEmbeddingModel, self).__init__()
        self.time_embedding = TimeEmbedding(embedding_dim)
        self.encoder = MLPTimeEncoder(embedding_dim, hidden_dims, output_dim)

    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_diff (torch.Tensor): Time difference from start in seconds. Shape: [batch_size, 1]
        Returns:
            torch.Tensor: Encoded time features. Shape: [batch_size, output_dim]
        """
        time_embedding = self.time_embedding(time_diff)
        encoded_time = self.encoder(time_embedding)
        return encoded_time

# Example Usage
if __name__ == "__main__":
    batch_size = 4
    time_differences = torch.tensor([[3600], [7200], [10800], [14400]], dtype=torch.float32)  # Shape [batch_size, 1]

    # Model Configuration
    embedding_dim = 32
    hidden_dims = [64, 128]
    output_dim = 64

    # Initialize Model
    model = TimeEmbeddingModel(embedding_dim, hidden_dims, output_dim)

    # Forward Pass
    time_embeddings = model(time_differences)
    print("Time Embeddings Shape:", time_embeddings.shape)
    print("Time Embeddings:", time_embeddings)
