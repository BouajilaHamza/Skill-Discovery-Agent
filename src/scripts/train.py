# src/scripts/train.py
import os
import yaml
import argparse
import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from collections import deque

from pytorch_lightning import Trainer, LightningModule
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
    """Collect a single rollout from the environment."""
    obs, _ = env.reset()
    skill = agent._sample_skill()
    episode_reward = 0
    episode_length = 0
    
    for _ in range(max_steps):
        # Convert observation to tensor and add batch dimension
        obs_tensor = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(agent.device)
        skill_tensor = torch.FloatTensor(skill).unsqueeze(0).to(agent.device)
        
        # Select action
        with torch.no_grad():
            action = agent.forward(obs_tensor, skill_tensor, deterministic=False)
            action = action.item()
        
        # Take step in environment
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
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        
        # Update observation
        obs = next_obs
        
        if done:
            break
    
    return episode_reward, episode_length

def train():
    args = parse_args()
    config = load_config(args.config)
    
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
    
    # Create agent
    agent = DIAYNAgent(config["agent"])
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"diayn_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="diayn",
        version=timestamp
    )
    
    # Set up model checkpointing
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="diayn-{epoch:03d}",
        save_top_k=3,
        monitor="train/avg_reward",
        mode="max",
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=config["training"]["max_episodes"],
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        enable_checkpointing=True,
        accelerator="cpu",
        devices=1,
        check_val_every_n_epoch=config["training"].get("eval_interval", 10),
    )
    
    # Training loop
    max_episodes = config["training"]["max_episodes"]
    eval_interval = config["training"].get("eval_interval", 10)
    
    # Metrics tracking
    episode_rewards = []
    episode_lengths = []
    
    # Progress bar
    pbar = tqdm(range(max_episodes), desc="Training")
    
    for episode in pbar:
        # Collect rollout
        episode_reward, episode_length = collect_rollout(env, agent)
        
        # Update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log metrics
        avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
        avg_length = np.mean(episode_lengths[-100:])
        
        # Log to TensorBoard
        trainer.logger.experiment.add_scalar("train/episode_reward", episode_reward, episode)
        trainer.logger.experiment.add_scalar("train/episode_length", episode_length, episode)
        trainer.logger.experiment.add_scalar("train/avg_reward", avg_reward, episode)
        trainer.logger.experiment.add_scalar("train/avg_length", avg_length, episode)
        
        # Update progress bar
        pbar.set_postfix({
            "reward": f"{episode_reward:.1f}",
            "avg_reward": f"{avg_reward:.1f}",
            "length": episode_length
        })
        
        # Perform evaluation
        if (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for _ in range(5):  # Run 5 evaluation episodes
                eval_reward, _ = collect_rollout(env, agent)
                eval_rewards.append(eval_reward)
            
            avg_eval_reward = np.mean(eval_rewards)
            trainer.logger.experiment.add_scalar("eval/avg_reward", avg_eval_reward, episode)
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "diayn_final.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")
    
    # Save training metrics
    metrics = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'config': config
    }
    
    metrics_path = os.path.join(log_dir, 'training_metrics.pt')
    torch.save(metrics, metrics_path)
    print(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()