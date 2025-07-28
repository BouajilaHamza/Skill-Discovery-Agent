import yaml
import argparse
import numpy as np
from typing import Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
from src.agents.diayn_agent import DIAYNAgent
from src.envs.minigrid_wrapper import MiniGridWrapper




Transition = namedtuple('Transition', 
                       ('state', 'action', 'skill', 'next_state', 'done', 'reward'))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diayn.yaml")
    parser.add_argument("--log_dir", type=str, default="logs")
    return parser.parse_args()




def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)





def collect_rollout(env: MiniGridWrapper, agent: DIAYNAgent, max_steps: int = 1000) -> Tuple[float, int]:
    """Collect a single rollout from the environment with optimized data transfer.
    Args:
        env: The environment to collect the rollout from
        agent: The agent to select actions
        max_steps: Maximum number of steps per episode
        
    Returns:
        Tuple containing total reward and episode length
    """
    obs, _ = env.reset()
    skill = agent.sample_skill()
    episode_reward = 0.0
    episode_length = 0
    done = False
    # print("Episode length: ", episode_length)
    # print("Max steps: ", max_steps)
    # print(episode_length < max_steps)
    while not done and episode_length < max_steps:
        
        action = agent.act(obs, skill)        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        transition = Transition(state=obs['observation'] if isinstance(obs, dict) and 'observation' in obs else obs,
                                action=action,
                                skill=skill,
                                next_state=next_obs['observation'] if isinstance(next_obs, dict) and 'observation' in next_obs else next_obs,
                                done=done,
                                reward=reward)
        agent.add_to_replay(transition)
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
    return episode_reward, episode_length








def evaluate(env: MiniGridWrapper, agent: DIAYNAgent, num_episodes: int = 5,episode: int = 0, writer: Optional[SummaryWriter] = None) -> float:
    """Evaluate the agent's performance."""
    agent.eval()
    eval_rewards = []
    
    for _ in range(num_episodes):
        episode_reward, _ = collect_rollout(env, agent)
        
        eval_rewards.append(episode_reward)
    avg_eval_reward = np.mean(eval_rewards)
    if writer:
        writer.add_scalar("eval/avg_reward", avg_eval_reward, episode)
    agent.train()
    return np.mean(eval_rewards)