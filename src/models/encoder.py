from src.models.base_model import BaseModel
from torch import nn
from typing import Tuple
import torch


class MiniGridEncoder(BaseModel):
    """Encoder for the MiniGrid environment So that it can be used in the DIAYN agent
    
    Args:
        input_shape (Tuple[int]): The shape of the input observation.
        hidden_size (int): The size of the hidden layer.
    
    Returns:
        torch.Tensor: The encoded observation.
    """
    
    def __init__(self, obs_shape: Tuple[int], feature_dim: int = 64, obs_type: str = "rgb"):
        super().__init__()
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        
        # Determine input channels based on observation type
        self.in_channels = 3 if obs_type == "rgb" else 1
        
        # CNN architecture for processing observations
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        

        with torch.no_grad():
            # Create a dummy input with correct shape (N, C, H, W)
            dummy = torch.zeros(1, self.in_channels, *obs_shape[:2])
            conv_out = self.conv(dummy)
            self.conv_output_dim = conv_out.shape[1]
            

        self.fc = nn.Linear(self.conv_output_dim, feature_dim)
        self.feature_dim = feature_dim
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder
        
        Args:
            obs (torch.Tensor): The input observation of shape (batch, H, W, C) or (H, W, C)
            
        Returns:
            torch.Tensor: The encoded observation of shape (batch, hidden_dim)
        """
        # Ensure we have a batch dimension
        if len(obs.shape) == 3:  # (H, W, C) -> (1, H, W, C)
            obs = obs.unsqueeze(0)
            
        # Convert to float and normalize if needed
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
            
        # Convert from NHWC to NCHW format expected by PyTorch
        if obs.shape[-1] in [1, 3]:  # If channels are last
            obs = obs.permute(0, 3, 1, 2)  # NHWC -> NCHW
            
        # Ensure we have the right number of channels
        if self.obs_type == 'rgb' and obs.shape[1] != 3:
            if obs.shape[1] == 1:  # If grayscale, repeat to 3 channels
                obs = obs.repeat(1, 3, 1, 1)
            else:
                raise ValueError(f"Expected 1 or 3 channels for RGB, got {obs.shape[1]} channels")
                

        x = self.conv(obs)
        return torch.relu(self.fc(x))
    

