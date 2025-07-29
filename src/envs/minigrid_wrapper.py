import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MiniGridWrapper(gym.Wrapper):
    """Wrapper for the MiniGrid environment"""
    def __init__(self,env,skill_dim=8,obs_type="rgb"):
        super().__init__(env)
        self.skill_dim = skill_dim
        self.obs_type = obs_type
        
        if obs_type == "rgb":
            self.obs_shape = (7,7,3)
        else: #obs_type = grid
            self.obs_shape = (7,7)
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0,high=255,
                shape=self.obs_shape,
                dtype=np.uint8
            ),
            "skill": spaces.Box(
                low=-1.0,high=1.0,
                shape=(skill_dim,),
                dtype=np.float32
            )
        })
    
    def reset(self,**kwargs):
        obs,info = super().reset(**kwargs)
        return self._process_obs(obs),info
    
    def step(self,action):
        obs,reward,terminated,truncated,info = self.env.step(action)
        return self._process_obs(obs),reward,terminated,truncated,info
    
    def _process_obs(self,obs):
        """
        Process the observation to match the observation space
        """
        
        if self.obs_type == "rgb":
            obs_array = obs["image"][...,:3]
        else:
            obs_array = obs["image"][...,0]
        
        skill = np.random.uniform(-1,1,size=(self.skill_dim,))
        
        return {
            "observation":obs_array,
            "skill":skill
        }
        

class ObservationExtractor(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['observation']
        
    def observation(self, obs):
        return obs['observation']
        