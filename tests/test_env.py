import gymnasium as gym
from src.envs import *  # This will register our custom environments

def test_environment(env_id):
    try:
        env = gym.make(env_id, render_mode="human")
        print(f"Successfully created environment: {env.spec.id}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test stepping through the environment
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs['image'].shape}")
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
        return True
    except Exception as e:
        print(f"Error creating environment {env_id}: {e}")
        return False

if __name__ == "__main__":
    env_id = "MiniGrid-Empty-8x8-v0"
    print(f"Testing environment: {env_id}")
    success = test_environment(env_id)
    
    if success:
        print("\nEnvironment test successful! You can now use this environment in your training script.")
    else:
        print("\nEnvironment test failed. Please check the error message above.")
