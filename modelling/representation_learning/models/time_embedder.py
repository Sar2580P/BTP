import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        """
        Logarithmic time embedding optimized for large time spans.
        
        Args:
            embedding_dim (int): Dimensionality of the time embedding.
            epsilon (float): Small value added to time to avoid log(0).
        """
        super(TimeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Create projection layers
        self.linear1 = nn.Linear(1, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_diff (torch.Tensor): Time difference from start in seconds. Shape: [batch_size, 1]
        Returns:
            torch.Tensor: Time embeddings. Shape: [batch_size, embedding_dim]
        """
        
        # Get embeddings from both raw time and log time
        embedding = self.linear1(time_diff)        
        # Apply normalization and activation
        embedding = self.norm(embedding)
        embedding = F.gelu(embedding)
        
        return embedding

# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 32
    embedding_dim = 64
    
    # Create embedding model
    time_embedder = TimeEmbedding(embedding_dim=embedding_dim)
    
    # Generate sample input at different scales
    test_times = torch.tensor([
        [0.0], [10.0], [20.0], [50.0], [100.0],
        [200.0], [500.0], [1000.0], [2000.0], [5000.0], [10000.0]
    ])
    
    # Get embeddings
    embeddings = time_embedder(test_times)
    print(f"Input times shape: {test_times.shape}")
    print(f"Output embeddings shape: {embeddings.shape}")