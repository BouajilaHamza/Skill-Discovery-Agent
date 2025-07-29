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
    parser.add_argument("--config", type=str, default="/configs/diayn.yaml")
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



def collect_parallel_rollout(env, agent: DIAYNAgent, max_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect a rollout using vectorized environments.
    
    Args:
        env: Vectorized environment (SyncVectorEnv or AsyncVectorEnv)
        agent: DIAYN agent
        max_steps: Maximum number of steps per episode
    
    Returns:
        Tuple of (total_rewards, episode_lengths) arrays per environment
    """
    num_envs = env.num_envs
    obs, _ = env.reset()
    skills = np.array([agent.sample_skill() for _ in range(num_envs)], dtype=np.float32)

    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)
    dones = np.zeros(num_envs, dtype=bool)

    for _ in range(max_steps):
        actions = []
        for i in range(num_envs):
            if not dones[i]:
                action = agent.act(obs[i], skills[i])
            else:
                action = 0  # Dummy action for already-done envs
            actions.append(action)

        actions = np.array(actions)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        step_dones = np.logical_or(terminations, truncations)

        for i in range(num_envs):
            if dones[i]:  # Skip if already done
                continue

            transition = Transition(
                state=obs[i]['observation'] if isinstance(obs[i], dict) else obs[i],
                action=actions[i],
                skill=skills[i],
                next_state=next_obs[i]['observation'] if isinstance(next_obs[i], dict) else next_obs[i],
                done=step_dones[i],
                reward=rewards[i]
            )
            agent.add_to_replay(transition)

            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1

        dones = np.logical_or(dones, step_dones)
        obs = next_obs

        if np.all(dones):
            break

    return episode_rewards, episode_lengths







def evaluate(env, agent: DIAYNAgent, num_episodes: int = 5, episode: int = 0, writer: Optional[SummaryWriter] = None) -> float:
    """Evaluate the agent's performance on a vectorized environment."""
    agent.eval()
    eval_rewards = []
    eval_lengths = []
    
    episodes_collected = 0
    num_envs = env.num_envs

    while episodes_collected < num_episodes:
        rewards, lengths = collect_parallel_rollout(env, agent, max_steps=1000)
        
        for r ,l in zip(rewards, lengths):
            eval_rewards.append(r)
            eval_lengths.append(l)
            episodes_collected += 1
            if episodes_collected >= num_episodes:
                break

    avg_eval_reward = np.mean(eval_rewards)
    if writer:
        writer.add_scalar("eval/avg_reward", avg_eval_reward, episode)
        writer.add_scalar("eval/avg_length", np.mean(eval_lengths), episode)
    
    agent.train()
    return avg_eval_reward
