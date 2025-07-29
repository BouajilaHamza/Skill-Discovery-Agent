import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from gymnasium.vector.sync_vector_env import SyncVectorEnv

from src.agents.diayn_agent import DIAYNAgent
from src.utils.train_utils import parse_args, load_config, evaluate
from src.utils.env_utils import make_env


def train():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    
    print(f"Using device: {device} | Device  Name: {device_name}")
    # seed = config.get("seed", 42)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    
    # Create  vectorized environments
    num_envs = config.get("training", {}).get("env_parallel", 1)
    env_thunks = [make_env(config["env_id"], config.get("obs_type", "rgb")) for _ in range(num_envs)]
    print(env_thunks[0])
    env = SyncVectorEnv(env_thunks) 

    sample_obs = env.reset()[0]  # env.reset() returns (obs, info)
    print(sample_obs.shape)
    config["agent"]["obs_shape"] = sample_obs.shape[1:]
    config["agent"]["action_dim"] = env.action_space.nvec[0]
    

    # Create unique log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"diayn_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer with the correct log directory
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {os.path.abspath(log_dir)}")


    agent = DIAYNAgent(config["agent"], writer=writer, log_dir=log_dir).to(device)
    agent.log_model_graph()

    training_cfg = config.get("training", {})
    print(training_cfg.keys())
    num_episodes = training_cfg.get("max_episodes", 1000)
    eval_interval = training_cfg.get("eval_interval", 100)
    save_interval = training_cfg.get("save_interval", 100)
    
    # Training loop
    # best_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training progress bar
    pbar = tqdm(range(1, num_episodes + 1), desc="Training")
    
    for episode in pbar:
        try:
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
        writer.add_scalar('train/replay_buffer_size', len(agent.replay_buffer), episode)
        writer.add_scalar('train/batch_size', agent.batch_size, episode)
        if len(agent.replay_buffer) >= agent.batch_size:
            try:
                batch = agent.sample_batch(agent.batch_size)
                if batch is None:
                    continue

                # Update both discriminator and policy
                loss_d, loss_p = agent.update(batch, episode)
                
                # if writer is not None:
                #     writer.add_scalar('train/discriminator_loss', loss_d, episode)
                #     writer.add_scalar('train/policy_loss', loss_p, episode)
                #     writer.add_scalar('train/replay_buffer_size', len(agent.replay_buffer), episode)

            except RuntimeError as e:
                print(f"Error during training step: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for all kernels to finish
            
        # Perform evaluation
        if (episode + 1) % eval_interval == 0:
            _avg_eval_reward = evaluate(env, agent, num_episodes=5, episode=episode, writer=writer)
        

        if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"diayn_episode_{episode+1}.ckpt")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.state_dict(),
                'optimizer_d_state_dict': agent.optimizer_d.state_dict(),
                'optimizer_p_state_dict': agent.optimizer_p.state_dict(),
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'config': config
            }, checkpoint_path)
    

    final_model_path = os.path.join(checkpoint_dir, "diayn_final.pt")
    torch.save({
        'model_state_dict': agent.state_dict(),
        'optimizer_d_state_dict': agent.optimizer_d.state_dict(),
        'optimizer_p_state_dict': agent.optimizer_p.state_dict(),
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