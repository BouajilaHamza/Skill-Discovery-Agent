#!/usr/bin/env python3
"""
Visualization script for DIAYN skill discovery in MiniGrid environments.

This script loads a trained DIAYN agent and visualizes the learned skills
by rolling out episodes with different skill vectors.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from tqdm import tqdm

from src.agents.diayn_agent import DIAYNAgent
from src.envs.minigrid_wrapper import MiniGridWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize learned skills from a trained DIAYN agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--env_id", type=str, default="MiniGrid-Empty-8x8-v0",
                        help="MiniGrid environment ID")
    parser.add_argument("--num_skills", type=int, default=8,
                        help="Number of skills to visualize")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to run per skill")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--save_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        choices=["human", "rgb_array", "ansi"],
                        help="Rendering mode")
    return parser.parse_args()

def load_agent(checkpoint_path, env):
    """Load a trained DIAYN agent from checkpoint."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Create agent with dummy config (will be overridden)
    config = {
        "obs_shape": env.observation_space["observation"].shape,
        "action_dim": env.action_space.n,
        "skill_dim": 8,  # Will be overridden from checkpoint
    }
    
    agent = DIAYNAgent(config)
    agent.load_state_dict(checkpoint['state_dict'])
    agent.eval()
    return agent

def visualize_skills(agent, env, args):
    """Visualize learned skills by rolling out episodes with different skills."""
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create one-hot skill vectors
    skills = torch.eye(agent.skill_dim, device=agent.device)
    
    # For each skill, run episodes and collect trajectories
    skill_data = {}
    
    for skill_idx in range(min(args.num_skills, agent.skill_dim)):
        skill = skills[skill_idx]
        skill_data[skill_idx] = []
        
        print(f"\nVisualizing skill {skill_idx}...")
        
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'frames': []
            }
            
            # Get initial frame
            if args.render_mode == "rgb_array":
                frame = env.render()
                episode['frames'].append(frame)
            
            for step in range(args.max_steps):
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs["observation"]).unsqueeze(0).to(agent.device)
                
                # Get action from agent
                with torch.no_grad():
                    action = agent.forward(obs_tensor, skill.unsqueeze(0), deterministic=True)
                    action = action.item()
                
                # Take step in environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                # Store transition
                episode['observations'].append(obs["observation"])
                episode['actions'].append(action)
                episode['rewards'].append(reward)
                
                # Render if needed
                if args.render_mode == "rgb_array":
                    frame = env.render()
                    episode['frames'].append(frame)
                
                # Update observation
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Save episode data
            skill_data[skill_idx].append(episode)
            
            # Save frames as video
            if episode['frames'] and args.render_mode == "rgb_array":
                save_frames(episode['frames'], args.save_dir, skill_idx, ep)
    
    # Generate and save skill visualization
    plot_skill_visualization(skill_data, args)

def save_frames(frames, save_dir, skill_idx, episode_idx):
    """Save frames as a GIF or video."""
    try:
        import imageio
        
        # Create filename
        filename = os.path.join(save_dir, f"skill_{skill_idx}_episode_{episode_idx}.gif")
        
        # Save as GIF
        imageio.mimsave(filename, frames, fps=5)
        print(f"Saved visualization to {filename}")
        
    except ImportError:
        print("Could not save GIF: imageio not installed. Install with: pip install imageio")

def plot_skill_visualization(skill_data, args):
    """Create and save a plot showing the performance of each skill."""
    # Calculate average rewards per skill
    skill_rewards = []
    skill_lengths = []
    
    for skill_idx, episodes in skill_data.items():
        rewards = [np.sum(episode['rewards']) for episode in episodes]
        lengths = [len(episode['rewards']) for episode in episodes]
        
        skill_rewards.append(np.mean(rewards))
        skill_lengths.append(np.mean(lengths))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot average rewards
    ax1.bar(range(len(skill_rewards)), skill_rewards)
    ax1.set_title(f'Average Reward per Skill (n={args.num_episodes} episodes)')
    ax1.set_xlabel('Skill Index')
    ax1.set_ylabel('Average Reward')
    
    # Plot episode lengths
    ax2.bar(range(len(skill_lengths)), skill_lengths)
    ax2.set_title('Average Episode Length per Skill')
    ax2.set_xlabel('Skill Index')
    ax2.set_ylabel('Average Length (steps)')
    
    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(args.save_dir, 'skill_performance.png')
    plt.savefig(fig_path)
    print(f"Saved skill performance plot to {fig_path}")
    plt.close()

def main():
    args = parse_args()
    
    # Create environment
    env = gym.make(args.env_id, render_mode=args.render_mode)
    env = MiniGridWrapper(
        env,
        skill_dim=args.num_skills,
        obs_type="rgb"  # Always use RGB for visualization
    )
    
    # Load agent
    agent = load_agent(args.checkpoint, env)
    
    # Visualize skills
    with torch.no_grad():
        visualize_skills(agent, env, args)
    
    env.close()

if __name__ == "__main__":
    main()
