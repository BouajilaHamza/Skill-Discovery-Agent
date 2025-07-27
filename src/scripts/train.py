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
    
    # Convert initial observation to tensor
    obs_array = np.asarray(obs["observation"], dtype=np.float32)
    
    for _ in range(max_steps):
        # Convert to tensor and move to device
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(agent.device)
        skill_tensor = torch.FloatTensor(skill).unsqueeze(0).to(agent.device)
        
        # Ensure tensors are contiguous
        obs_tensor = obs_tensor.contiguous()
        skill_tensor = skill_tensor.contiguous()
        
        # Get action from agent
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=agent.device.type == 'cuda'):
            action = agent.act(obs_tensor, skill_tensor, deterministic=False)
        
        # Step environment
        next_obs, reward, done, _, _ = env.step(action.item() if torch.is_tensor(action) else action)
        
        # Convert next observation to numpy
        next_obs_array = np.asarray(next_obs["observation"], dtype=np.float32)
        
        # Store transition
        agent.add_to_replay(
            obs_array,  # Use numpy array
            action.item() if torch.is_tensor(action) else action,
            skill,
            next_obs_array,  # Use numpy array
            done,
            reward
        )
        
        # Update state
        obs_array = next_obs_array
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        if done:
            break
            
        # Clear CUDA cache periodically
        if episode_length % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return episode_reward, episode_length

def train():
    args = parse_args()
    config = load_config(args.config)
    
    # Set up device and mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up mixed precision
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
    
    # Initialize agent with config
    agent = DIAYNAgent(config).to(device)
    
    agent.train()
    
    # Enable optimizations if using CUDA
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")


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
    
    # Get training config
    max_episodes = config["training"]["max_episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    log_interval = config["training"].get("log_interval", 10)
    save_interval = config["training"].get("save_interval", 100)
    
    # Create progress bar
    pbar = tqdm(range(1, max_episodes + 1), desc="Training", unit="episode")
    
    # Initialize optimizers
    optimizers = agent.configure_optimizers()
    if isinstance(optimizers, (list, tuple)) and len(optimizers) > 0:
        if isinstance(optimizers[0], (list, tuple)):
            # Handle case where optimizers is a list of optimizers and schedulers
            optimizers = optimizers[0]
    
    # Unpack optimizers if we have multiple
    if isinstance(optimizers, (list, tuple)) and len(optimizers) >= 2:
        optimizer_d, optimizer_p = optimizers[0], optimizers[1]
    else:
        # Fallback to single optimizer if needed
        optimizer_d = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        optimizer_p = optimizer_d
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    best_reward = -float('inf')
    
    for episode in pbar:
        try:
            # Set models to train mode
            agent.train()
            
            # Collect rollout
            episode_reward, episode_length = collect_rollout(env, agent, max_steps)
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            total_steps += episode_length
            
            # Calculate running averages
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'length': episode_length,
                'steps': total_steps
            })
            
            # Log metrics
            if logger is not None:
                logger.experiment.add_scalar("train/episode_reward", episode_reward, episode)
                logger.experiment.add_scalar("train/episode_length", episode_length, episode)
                logger.experiment.add_scalar("train/avg_reward", avg_reward, episode)
                logger.experiment.add_scalar("train/avg_length", avg_length, episode)
                logger.experiment.add_scalar("train/total_steps", total_steps, episode)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                if logger is not None and hasattr(logger, 'save_checkpoint'):
                    checkpoint = {
                        'episode': episode,
                        'model_state_dict': agent.state_dict(),
                        'reward': episode_reward,
                        'optimizer_state_dict': [opt.state_dict() for opt in optimizers] if isinstance(optimizers, list) else optimizers.state_dict(),
                    }
                    logger.save_checkpoint(checkpoint, is_best=True)
            
            # Save model at intervals
            if episode % save_interval == 0 and logger is not None and hasattr(logger, 'save_checkpoint'):
                checkpoint = {
                    'episode': episode,
                    'model_state_dict': agent.state_dict(),
                    'reward': episode_reward,
                    'optimizer_state_dict': [opt.state_dict() for opt in optimizers] if isinstance(optimizers, list) else optimizers.state_dict(),
                }
                logger.save_checkpoint(checkpoint, is_best=False, filename=f'checkpoint_ep{episode}.pt')
            
            # Clear CUDA cache periodically
            if episode % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error during episode {episode}: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        
        # Perform training step
        if len(agent.replay_buffer) >= agent.batch_size:
            try:
                # Train discriminator
                optimizer_d.zero_grad()
                with torch.cuda.amp.autocast(enabled=agent.device.type == 'cuda'):
                    loss_d = agent.training_step(None, episode, optimizer_idx=0)
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
                
                # Train policy
                optimizer_p.zero_grad()
                with torch.cuda.amp.autocast(enabled=agent.device.type == 'cuda'):
                    loss_p = agent.training_step(None, episode, optimizer_idx=1)
                scaler.scale(loss_p).backward()
                scaler.step(optimizer_p)
                
                # Update scaler for next iteration
                scaler.update()
                
                # Log training metrics
                if logger is not None:
                    logger.experiment.add_scalar("train/loss_discriminator", loss_d.item(), episode)
                    logger.experiment.add_scalar("train/loss_policy", loss_p.item(), episode)
                    
            except RuntimeError as e:
                print(f"Error during training step: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all kernels to finish
            
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