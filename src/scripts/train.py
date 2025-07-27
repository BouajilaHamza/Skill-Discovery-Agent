import os
import yaml
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.agents.diayn_agent import DIAYNAgent
from src.envs.minigrid_wrapper import MiniGridWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diayn.yaml")
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def collect_rollout(env, agent, max_steps=1000):
    """Collect a single rollout from the environment with optimized data transfer."""
    obs, _ = env.reset()
    skill = agent._sample_skill()
    episode_reward = 0
    episode_length = 0
    
    # Pre-allocate tensors on GPU if available
    if torch.cuda.is_available():
        obs_buffer = torch.empty((1, *obs["observation"].shape), 
                               dtype=torch.float32, 
                               device=agent.device,
                               pin_memory=True)
        skill_buffer = torch.empty((1, len(skill)), 
                                 dtype=torch.float32, 
                                 device=agent.device,
                                 pin_memory=True)
    
    for _ in range(max_steps):
        # Efficient data transfer using pinned memory
        if torch.cuda.is_available():
            obs_buffer.copy_(torch.as_tensor(obs["observation"][None, ...], 
                                          device='cpu'), 
                          non_blocking=True)
            skill_buffer.copy_(torch.as_tensor(skill[None, ...], 
                                            device='cpu'), 
                            non_blocking=True)
            obs_tensor = obs_buffer
            skill_tensor = skill_buffer
        else:
            obs_tensor = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(agent.device)
            skill_tensor = torch.FloatTensor(skill).unsqueeze(0).to(agent.device)
        

        with torch.no_grad():
            action = agent.forward(obs_tensor, skill_tensor, deterministic=False)
            action = action.item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition in replay buffer
        agent.add_to_replay(
            obs["observation"],
            action,
            skill,
            next_obs["observation"],
            done,
            reward
        )
        
        episode_reward += reward
        episode_length += 1
        
        obs = next_obs
        
        if done:
            break
    
    return episode_reward, episode_length

def train():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up device and mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Create environment
    env = gym.make(config["env_id"], render_mode="rgb_array")
    env = MiniGridWrapper(
        env, 
        skill_dim=config["agent"]["skill_dim"],
        obs_type=config["obs_type"]
    )
    
    # Update obs_shape in config
    config["agent"]["obs_shape"] = env.observation_space["observation"].shape
    config["agent"]["action_dim"] = env.action_space.n
    
    # Initialize agent with config and move to device
    agent = DIAYNAgent(config).to(device)
    agent.train()
    
    # Enable cuDNN benchmarking for optimal performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"diayn_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="diayn",
        version=timestamp
    )
        

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diayn-{epoch:03d}",
        save_top_k=3,
        monitor="train/avg_reward",
        mode="max",
    )
    
    max_episodes = config["training"]["max_episodes"]
    eval_interval = config["training"].get("eval_interval", 10)
    save_interval = config["training"].get("save_interval", 100)

    episode_rewards = []
    episode_lengths = []
    
    pbar = tqdm(range(max_episodes), desc="Training")
    
    optimizer_d, optimizer_p = agent.configure_optimizers()
    
    for episode in pbar:
        agent.train()        
        episode_reward, episode_length = collect_rollout(env, agent)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        avg_reward = np.mean(episode_rewards[-100:])
        avg_length = np.mean(episode_lengths[-100:])
        
        # Log to TensorBoard if logger is available
        if logger:
            logger.experiment.add_scalar("train/episode_reward", episode_reward, episode)
            logger.experiment.add_scalar("train/episode_length", episode_length, episode)
            logger.experiment.add_scalar("train/avg_reward", avg_reward, episode)
            logger.experiment.add_scalar("train/avg_length", avg_length, episode)
        
        pbar.set_postfix({
            "reward": f"{episode_reward:.1f}",
            "avg_reward": f"{avg_reward:.1f}",
            "length": episode_length
        })
        
        # Perform training step
        if len(agent.replay_buffer) >= agent.batch_size:
            # Train discriminator
            optimizer_d.zero_grad()
            loss_d = agent.training_step(None, episode, optimizer_idx=0)
            loss_d.backward()
            optimizer_d.step()
            
            # Train policy
            optimizer_p.zero_grad()
            loss_p = agent.training_step(None, episode, optimizer_idx=1)
            loss_p.backward()
            optimizer_p.step()
            
            if logger:
                logger.experiment.add_scalar("train/loss_discriminator", loss_d.item(), episode)
                logger.experiment.add_scalar("train/loss_policy", loss_p.item(), episode)
        
        # Perform evaluation
        if (episode + 1) % eval_interval == 0:
            agent.eval()
            eval_rewards = []
            with torch.no_grad():
                for _ in range(5):
                    eval_reward, _ = collect_rollout(env, agent)
                    eval_rewards.append(eval_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            if logger:
                logger.experiment.add_scalar("eval/avg_reward", avg_eval_reward, episode)
            
            agent.train()
        

        if (episode + 1) % save_interval == 0 or episode == max_episodes - 1:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"diayn_episode_{episode+1}.ckpt")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'optimizer_p_state_dict': optimizer_p.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'config': config
            }, checkpoint_path)
    

    final_model_path = os.path.join(checkpoint_dir, "diayn_final.pt")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'config': config
    }, final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    

    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'config': config
    }
    
    metrics_path = os.path.join(log_dir, 'training_metrics.pt')
    torch.save(metrics, metrics_path)
    print(f"Training metrics saved to {metrics_path}")
    
    return agent

if __name__ == "__main__":
    train()