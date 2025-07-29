from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseAgent(nn.Module, ABC):
    """Base class for all agents.
    
    This base class provides common functionality for all agents, including
    device management and basic training utilities.
    """
    
    def __init__(self, config: Dict[str, Any], writer: Optional[SummaryWriter] = None):
        """Initialize the base agent.
        
        Args:
            config: Configuration dictionary for the agent
            writer: TensorBoard SummaryWriter for logging
        """
        super().__init__()
        self.config = config
        self.writer = writer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        """Select an action given an observation.
        
        Args:
            obs: Observation from the environment
            deterministic: Whether to sample deterministically
            
        Returns:
            Action to take
        """
        pass
    
    def to(self, device):
        """Move model to device and update self.device."""
        self.device = device
        return super().to(device)
        
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard.
        
        Args:
            tag: Tag for the scalar
            value: Value to log
            step: Current step for x-axis
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram to TensorBoard.
        
        Args:
            tag: Tag for the histogram
            values: Values to create histogram from
            step: Current step for x-axis
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: nn.Module, sample_input: Any):
        """Log model graph to TensorBoard.
        
        Args:
            model: Model to log
            sample_input: Sample input for the model
        """
        if self.writer is not None:
            try:
                self.writer.add_graph(model, sample_input)
                self.writer.flush()
            except Exception as e:
                print(f"Failed to log model graph: {e}")
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint to
            **kwargs: Additional items to save in checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            **kwargs
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, **kwargs):
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            **kwargs: Additional arguments to pass to model constructor
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = {**checkpoint['config'], **kwargs}
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model