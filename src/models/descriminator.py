import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel


class SkillDiscriminator(BaseModel):
    """Skill discriminator for DIAYN that classifies which skill was used."""
    
    def __init__(self, input_dim: int, skill_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def compute_reward(self, state: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Compute the intrinsic reward for the given state and skill."""
        with torch.no_grad():
            logits = self.forward(state)
            log_probs = F.log_softmax(logits, dim=-1)
            return (log_probs * skill).sum(dim=-1, keepdim=True)
