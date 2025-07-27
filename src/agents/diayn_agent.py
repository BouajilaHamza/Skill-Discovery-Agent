import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, Dict, Any

from src.agents.base_agent import BaseAgent
from src.models.base_model import BaseModel

# Define transition tuple
Transition = namedtuple('Transition', 
                       ('state', 'action', 'skill', 'next_state', 'done', 'reward'))

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
    




class SkillDiscriminator(BaseModel):
    """
    Skill discriminator for the DIAYN agent
    
    Args:
        input_dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
    
    Returns:
        torch.Tensor: The skill discriminator output.
    """
    
    def __init__(self,state_dim,skill_dim,hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,skill_dim)
        )
        
    def forward(self,state:torch.Tensor):
        return self.net(state)
    
    
    def compute_reward(self,state,skill):
        logits = self(state)
        log_probs = F.log_softmax(logits,dim=-1)
        return (log_probs * skill).sum(dim=-1) + np.log(skill.size(1))



class DIAYNAgent(BaseAgent):
    """DIAYN agent for the MiniGrid environment"""
    def __init__(self,config:Dict[str,Any]):
        super().__init__(config)
        
        #Environment parameters
        self.obs_shape = config["agent"]["obs_shape"]
        self.action_dim = config["agent"]["action_dim"]
        self.skill_dim = config.get("skill_dim",8)
        self.obs_type = config.get("obs_type","rgb")
        
        # Training parameters
        self.lr = float(config.get("lr", 3e-4))
        self.gamma = float(config.get("gamma", 0.99))
        self.entropy_coeff = float(config.get("entropy_coeff", 0.01))
        self.batch_size = int(config.get("batch_size", 64))
        self.replay_size = int(config.get("replay_size", 10000))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        
        #Models
        self.encoder = MiniGridEncoder(self.obs_shape,
                                       feature_dim = config.get("hidden_dim",64),
                                       obs_type = self.obs_type
                                       ).to(self.device)
        
        #Policy Network 
        self.policy = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + self.skill_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,self.action_dim)
        ).to(self.device)
        
        #Discriminator Network
        self.discriminator = SkillDiscriminator(
            self.encoder.feature_dim,
            self.skill_dim,
            hidden_dim = config.get("hidden_dim",64)
        ).to(self.device)
        
        #Replay Buffer
        self.replay_buffer = deque(maxlen=self.replay_size)
        
        #Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def forward(self,obs:torch.Tensor,skill:torch.Tensor,deterministic:bool=False) -> torch.Tensor:
        """
        Forward pass of the agent
        Args: 
            obs (torch.Tensor): The observation.
            skill (torch.Tensor): The skill.
        Returns:
            torch.Tensor: The action.
        """
        with torch.no_grad():
            encoded_obs = self.encoder(obs) #return the state in latent space
            x = torch.cat([encoded_obs,skill],dim=-1).to(self.device)
            logits = self.policy(x).to(self.device)
            if deterministic:
                return torch.argmax(logits,dim=-1).to(self.device)
            else:
                probs = F.softmax(logits,dim=-1).to(self.device)
                return torch.multinomial(probs,1).squeeze(-1).to(self.device)
            
    
    def act(self,obs:torch.Tensor,skill:torch.Tensor=None,deterministic:bool=False) -> torch.Tensor:
        """Select an action given the observation and skill"""
        if skill is None:
            skill = self._sample_skill()
        skill = torch.FloatTensor(skill).unsqueeze(0).to(self.device) if not isinstance(skill,torch.Tensor) else skill
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device) if not isinstance(obs,torch.Tensor) else obs
        return self.forward(obs,skill,deterministic).cpu().numpy().item()
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Perform a single training step with mixed precision support.
        
        Args:
            batch: Batch of transitions
            batch_idx: Batch index
            optimizer_idx: Index of the optimizer (0: discriminator, 1: policy)
            
        Returns:
            dict: Dictionary containing loss and other metrics
        """
        states, actions, skills, next_states, dones, rewards = self._unpack_batch(batch)
        
        with amp.autocast(enabled=self.device.type == 'cuda'):
            # Encode states and next_states
            with torch.no_grad():
                states_enc = self.encoder(states)
                next_states_enc = self.encoder(next_states)
            
            # Train Discriminator
            if optimizer_idx == 0:
                logits = self.discriminator(next_states_enc)
                loss_d = F.cross_entropy(logits, skills.argmax(dim=-1))
                self.log("train/loss_discriminator", loss_d, prog_bar=True)
                return loss_d
                
            # Compute Policy
            if optimizer_idx == 1:
                policy_input = torch.cat([states_enc, skills], dim=-1)
                logits = self.policy(policy_input)
                
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                
                with torch.no_grad():
                    intrinsic_reward = self.discriminator.compute_reward(next_states_enc, skills)
                    
                # Compute policy gradient loss
                policy_loss = -(log_probs.gather(1, actions.unsqueeze(1)) * intrinsic_reward.detach()).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                total_loss = policy_loss + entropy_loss
                
                # Log metrics
                self.log("train/loss_policy", policy_loss, prog_bar=True)
                self.log("train/entropy", entropy.mean(), prog_bar=True)
                self.log("train/total_loss", total_loss)
                self.log("train/avg_intrinsic_reward", intrinsic_reward.mean())
                
                return total_loss
            
    def configure_optimizers(self):
        """Configure optimizers for discriminator and policy with weight decay."""
        # Use AdamW with weight decay
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), 
            lr=self.lr,
            weight_decay=1e-5,
            eps=1e-5
        )
        opt_p = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
            weight_decay=1e-5,
            eps=1e-5
        )
        
        # Learning rate scheduling
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=1000)
        scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=1000)
        
        return [opt_d, opt_p], [scheduler_d, scheduler_p]   
        
    def _sample_batch(self):
        """Sample a batch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            batch_size = len(self.replay_buffer)
        else:
            batch_size = self.batch_size
            
        # Sample random indices
        indices = np.random.choice(len(self.replay_buffer), size=batch_size, replace=False)
        transitions = [self.replay_buffer[i] for i in indices]
        batch = Transition(*zip(*transitions))
        
        return (
            torch.stack(batch.state).to(self.device),
            torch.cat(batch.action).to(self.device),
            torch.stack(batch.skill).to(self.device),
            torch.stack(batch.next_state).to(self.device),
            torch.stack(batch.done).to(self.device),
            torch.stack(batch.reward).to(self.device)
        )
    
    
    def _sample_skill(self):
        """Sample a random one-hot skill vector"""
        skill = np.zeros(self.skill_dim)
        skill[np.random.randint(self.skill_dim)] = 1
        return skill

    
    def add_to_replay(self,state,action,skill,next_state,done,reward):
        
        """Add transition to replay buffer."""
        self.replay_buffer.append(Transition(
            torch.FloatTensor(state).to(self.device),
            torch.LongTensor([action]).to(self.device),
            torch.FloatTensor(skill).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor([done]).to(self.device),
            torch.FloatTensor([reward]).to(self.device)
        ))
        
    

        