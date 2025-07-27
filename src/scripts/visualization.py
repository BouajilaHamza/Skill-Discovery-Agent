"""Visualization utilities for the Skill Discovery Agent.

This module provides functions to visualize training metrics, agent behavior,
and skill discovery progress.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, Any
import imageio
import pickle

def plot_training_metrics(metrics: Dict[str, Any], output_dir: str) -> None:
    """Plot and save training metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    episode_rewards = metrics.get('episode_rewards', [])
    episode_lengths = metrics.get('episode_lengths', [])
    
    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_rewards.png'))
    plt.close()
    
    # Plot moving average of rewards
    if len(episode_rewards) > 100:
        window_size = min(100, len(episode_rewards) // 10)
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        
        plt.figure(figsize=(12, 6))
        plt.plot(moving_avg)
        plt.title(f'Moving Average Reward (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'moving_avg_rewards.png'))
        plt.close()
    
    # Plot episode lengths
    if episode_lengths:
        plt.figure(figsize=(12, 6))
        plt.plot(episode_lengths)
        plt.title('Episode Lengths Over Time')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_lengths.png'))
        plt.close()


def plot_skill_usage(skills_used: np.ndarray, output_dir: str) -> None:
    """Plot skill usage statistics.
    
    Args:
        skills_used: Array of skill indices used during training
        output_dir: Directory to save the plots
    """
    if len(skills_used) == 0:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot skill distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=skills_used)
    plt.title('Skill Usage Distribution')
    plt.xlabel('Skill Index')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'skill_distribution.png'))
    plt.close()


def save_episode_as_gif(frames: list, output_path: str, duration: float = 0.1) -> None:
    """Save a sequence of frames as a GIF.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Path to save the GIF
        duration: Duration between frames in seconds
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, duration=duration, loop=0)


def load_metrics_safely(metrics_path: str) -> dict:
    """Safely load metrics with PyTorch 2.6+ compatibility."""
    from torch.serialization import add_safe_globals
    
    # Add numpy's scalar type to safe globals for loading
    add_safe_globals([np._core.multiarray.scalar])
    
    try:
        # First try with weights_only=True (safer)
        return torch.load(metrics_path, 
                       map_location=torch.device('cpu'),
                       weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        print(f"Falling back to less secure loading method: {e}")
        # Fall back to weights_only=False if needed (less secure)
        return torch.load(metrics_path,
                       map_location=torch.device('cpu'),
                       weights_only=False)

def visualize_training_run(log_dir: str) -> None:
    """Generate all visualizations for a training run.
    
    Args:
        log_dir: Directory containing training logs and metrics
    """
    metrics_path = os.path.join(log_dir, 'training_metrics.pt')
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return
    
    # Load metrics safely
    metrics = load_metrics_safely(metrics_path)
    
    # Create visualizations directory
    viz_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate plots
    plot_training_metrics(metrics, viz_dir)
    
    # Plot skill usage if available
    if 'skills_used' in metrics:
        plot_skill_usage(metrics['skills_used'], viz_dir)
    
    print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing training logs')
    
    args = parser.parse_args()
    visualize_training_run(args.log_dir)
