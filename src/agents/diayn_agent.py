import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, Any, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from src.agents.base_agent import BaseAgent
from src.models.minigrid_model import MiniGridEncoder
from src.models.descriminator_model import SkillDiscriminator


Transition = namedtuple('Transition', 
                       ('state', 'action', 'skill', 'next_state', 'done', 'reward'))


class DIAYNAgent(BaseAgent):
    """Diversity is All You Need (DIAYN) agent implementation."""
    
    def __init__(self, config: Dict[str, Any], writer: Optional[SummaryWriter] = None):
        """Initialize the DIAYN agent.
        
        Args:
            config: Configuration dictionary containing agent parameters
            writer: TensorBoard SummaryWriter for logging
        """
        super().__init__(config, writer)
        
        # Environment parameters
        self.obs_shape = config["obs_shape"]
        self.action_dim = config["action_dim"]
        self.skill_dim = config.get("skill_dim", 8)
        
        # Training parameters
        self.lr = float(config.get("lr", 3e-4))
        self.gamma = float(config.get("gamma", 0.99))
        self.batch_size = int(config.get("batch_size", 64))
        self.replay_size = int(config.get("replay_size", 10000))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        
        # Initialize models
        self.encoder = MiniGridEncoder(
            self.obs_shape,
            feature_dim=config.get("hidden_dim", 64)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + self.skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
        
        # Discriminator network
        self.discriminator = SkillDiscriminator(
            state_dim=self.encoder.feature_dim,
            skill_dim=self.skill_dim,
            hidden_dim=config.get("hidden_dim", 256)
        )
        
        # Optimizers
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        self.optimizer_p = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr,
            weight_decay=1e-5
        )
        
        # Learning rate schedulers
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=1000
        )
        self.scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_p, T_max=1000
        )
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.replay_size)
        
        # Move to device
        self.to(self.device)
    
    
    
    
    def forward(self, obs: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Forward pass through the agent."""
        encoded = self.encoder(obs)
        x = torch.cat([encoded, skill], dim=-1)
        return self.policy(x)
    
    
    
    
    def act(self, obs: Dict[str, Any], skill: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action given an observation and skill.
        Args:
            obs: Observation from the environment
            skill: Skill vector
            deterministic: Whether to sample deterministically
            
        Returns:
            Action to take
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(self.device)
            skill_tensor = torch.FloatTensor(skill).unsqueeze(0).to(self.device)
            logits = self.forward(obs_tensor, skill_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, num_samples=1)
                
            return action.item()
    
    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Tuple[float, float]:
        """Update the agent's parameters using a batch of experiences.
        
        Args:
            batch: Batch of transitions
            step: Current training step (for logging)
            
        Returns:
            Tuple of (discriminator_loss, policy_loss)
        """
        # Unpack batch
        states = batch["state"].to(self.device)
        actions = batch["action"].to(self.device)
        skills = batch["skill"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].to(self.device)
        
        # Train discriminator
        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            # Encode states
            states_enc = self.encoder(states)
            next_states_enc = self.encoder(next_states).detach()
            
            # Train discriminator
            logits = self.discriminator(next_states_enc)
            loss_d = F.cross_entropy(logits, skills.argmax(dim=-1))
            
            # Train policy
            policy_input = torch.cat([states_enc, skills], dim=-1)
            logits = self.policy(policy_input)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            
            # Compute intrinsic reward
            with torch.no_grad():
                pred_skill_probs = F.softmax(self.discriminator(next_states_enc), dim=-1)
                log_pred_skill_probs = torch.log(pred_skill_probs + 1e-6)
                intrinsic_reward = (log_pred_skill_probs * skills).sum(dim=-1)
                
            # Compute policy loss
            policy_loss = -(log_probs.gather(1, actions.unsqueeze(1)) * intrinsic_reward.unsqueeze(1)).mean()
            entropy_loss = -self.entropy_coef * entropy.mean()
            loss_p = policy_loss + entropy_loss
            
        # Update discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
        self.optimizer_d.step()
        
        # Update policy
        self.optimizer_p.zero_grad()
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.policy.parameters()),
            0.5
        )
        self.optimizer_p.step()
        
        # Update learning rates
        self.scheduler_d.step()
        self.scheduler_p.step()
        
        # Log metrics
        if self.writer is not None:
            self.writer.add_scalar('train/discriminator_loss', loss_d.item(), step)
            self.writer.add_scalar('train/policy_loss', loss_p.item(), step)
            self.writer.add_scalar('train/entropy', entropy.mean().item(), step)
            self.writer.add_scalar('train/intrinsic_reward', intrinsic_reward.mean().item(), step)
            
            # Log learning rates
            self.writer.add_scalar('lr/discriminator', self.scheduler_d.get_last_lr()[0], step)
            self.writer.add_scalar('lr/policy', self.scheduler_p.get_last_lr()[0], step)
        
        return loss_d.item(), loss_p.item()
    
    def add_to_replay(self, transition: Transition) -> None:
        """Add a transition to the replay buffer."""
        self.replay_buffer.append(transition)
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Dictionary containing batched tensors for states, actions, skills, next_states, dones, rewards.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        transitions = random.sample(self.replay_buffer, batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack([torch.FloatTensor(s) for s in batch.state])
        actions = torch.LongTensor(batch.action)
        skills = torch.FloatTensor(np.array(batch.skill))
        next_states = torch.stack([torch.FloatTensor(s) for s in batch.next_state])
        dones = torch.FloatTensor(batch.done)
        rewards = torch.FloatTensor(batch.reward)

        return {
            'state': states,
            'action': actions,
            'skill': skills,
            'next_state': next_states,
            'done': dones,
            'reward': rewards
        }
    
    def save_checkpoint(self, path: str) -> None:
        """Save agent state to checkpoint."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'optimizer_p_state_dict': self.optimizer_p.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'scheduler_p_state_dict': self.scheduler_p.state_dict(),
            'replay_buffer': self.replay_buffer,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load agent state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_p.load_state_dict(checkpoint['optimizer_p_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        self.scheduler_p.load_state_dict(checkpoint['scheduler_p_state_dict'])
        self.replay_buffer = checkpoint['replay_buffer']
        self.config = checkpoint['config']
        
        # Move models to device
        self.to(self.device)
    
    def log_model_graph(self) -> None:
        """Log the model graph to TensorBoard."""
        if self.writer is not None:
            try:
                # Create dummy inputs
                dummy_obs = torch.zeros(1, *self.obs_shape).to(self.device)
                dummy_skill = torch.zeros(1, self.skill_dim).to(self.device)
                
                # Create a wrapper module for the forward pass
                class ForwardWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, obs, skill):
                        return self.model(obs, skill)
                
                # Add graph to TensorBoard
                self.writer.add_graph(
                    ForwardWrapper(self),
                    (dummy_obs, dummy_skill)
                )
                self.writer.flush()
            except Exception as e:
                print(f"Failed to log model graph: {e}")
    
    def sample_skill(self) -> np.ndarray:
        """Sample a random one-hot skill vector."""
        skill_idx = np.random.randint(0, self.skill_dim)
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[skill_idx] = 1.0
        return skill
    
    def train(self, mode: bool = True) -> 'DIAYNAgent':
        """Set the agent in training mode."""
        super().train(mode)
        self.encoder.train(mode)
        self.policy.train(mode)
        self.discriminator.train(mode)
        return self
    
    def eval(self) -> 'DIAYNAgent':
        """Set the agent in evaluation mode."""
        return self.train(False)
    
    def to(self, device) -> 'DIAYNAgent':
        """Move the agent to the specified device."""
        super().to(device)
        self.encoder = self.encoder.to(device)
        self.policy = self.policy.to(device)
        self.discriminator = self.discriminator.to(device)
        return self
    
    def store_transition(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)

