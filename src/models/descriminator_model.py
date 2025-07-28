import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.base_model import BaseModel



class SkillDiscriminator(nn.Module):
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


    
    def _unpack_batch(self, batch):
        """Unpack a batch of transitions from the replay buffer.
        
        Args:
            batch: Batch of transitions. Can be:
                - List of transitions (state, action, skill, next_state, done, reward)
                - Tuple of (states, actions, skills, next_states, dones, rewards)
                
        Returns:
            tuple: (states, actions, skills, next_states, dones, rewards) as torch tensors
        """
        # Handle case where batch is already a tuple of tensors
        if isinstance(batch, (list, tuple)) and len(batch) == 6 and all(torch.is_tensor(x) for x in batch):
            return batch
            
        # Handle case where batch is a list of transitions
        if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], (list, tuple)):
            # Transpose the batch: from list of transitions to list of fields
            states, actions, skills, next_states, dones, rewards = zip(*batch)
        else:
            # Assume batch is already in the correct format
            states, actions, skills, next_states, dones, rewards = batch
        
        # Convert to numpy arrays if they aren't already
        states = np.array(states) if not isinstance(states, np.ndarray) else states
        next_states = np.array(next_states) if not isinstance(next_states, np.ndarray) else next_states
        skills = np.array(skills) if not isinstance(skills, np.ndarray) else skills
        
        # Convert to tensors and move to device
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        skills = torch.as_tensor(skills, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        
        # Ensure correct shapes
        if len(states.shape) == 3:  # (B, H, W) -> (B, 1, H, W)
            states = states.unsqueeze(1)
        if len(next_states.shape) == 3:
            next_states = next_states.unsqueeze(1)
            
        return states, actions, skills, next_states, dones, rewards
        
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
        
    
    def log_model_graph(self):
        """Log the model graph to TensorBoard"""
        if self.writer is not None:
            try:
                # Create sample inputs
                sample_obs = torch.zeros(1, *self.obs_shape, device=self.device)
                sample_skill = torch.zeros(1, self.skill_dim, device=self.device)
                
                # Create a simple forward function
                def forward(obs, skill):
                    with torch.no_grad():
                        encoded = self.encoder(obs)
                        x = torch.cat([encoded, skill], dim=-1)
                        return self.policy(x)
                
                # Add the graph
                self.writer.add_graph(forward, (sample_obs, sample_skill))
                self.writer.flush()
                print("Successfully saved model graph to TensorBoard")
            except Exception as e:
                print(f"Could not save model graph: {e}")
                import traceback
                traceback.print_exc()





