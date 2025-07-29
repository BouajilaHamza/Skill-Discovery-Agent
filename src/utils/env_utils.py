from src.envs.minigrid_wrapper import MiniGridWrapper,ObservationExtractor

import gymnasium as gym

def make_env(env_id, obs_type):
    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = MiniGridWrapper(env, obs_type=obs_type)
        env = ObservationExtractor(env)
        return env
    return _thunk

