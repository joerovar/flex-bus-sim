from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rl_env import FlexSimEnv
import pandas as pd
from itertools import product
import numpy as np

# function to train a simple PPO agent on the FlexSim environment
def train_flexsim(reward_weights, n_steps=64, total_timesteps=1200, 
                  verbose=0, save=False, test=False, 
                  learning_rate=0.0003, gamma=0.99, clip_range=0.2):
    env = Monitor(FlexSimEnv(reward_weights=reward_weights))
    
    env.reset()

    # Initialize the PPO agent with specified n_steps and verbosity
    model = PPO("MultiInputPolicy", env, verbose=verbose, n_steps=n_steps, learning_rate=learning_rate,
                gamma=gamma, clip_range=clip_range)
    
    # Train the agent for the specified number of timesteps
    model.learn(total_timesteps=total_timesteps)

    if test:
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
        print(f"Mean reward: {round(mean_reward, 3)} +/- {round(std_reward, 3)}")

    # Save the agent if save=True
    if save:
        weight_off_trips = str(int(reward_weights['off_schedule_trips']*-10))
        model_path = "models/ppo_" + weight_off_trips
        model.save(model_path)
        print(f"Agent saved in {model_path}.")
    print('------')

def grid_search_flexsim(
    lr_values=[0.0005, 0.0006],# learning rate
    ts_values=[24000, 28000],# timesteps
    gamma_values=[0.98, 0.99],# discount factor
    clip_values=[0.2],
    n_steps_values=[128, 256],
    verbose=0
):  
    # Dictionary to store results
    results = {
        'lr': [],
        'ts': [],
        'gamma': [],
        'clip': [],
        'n_steps': [],
        'mean_reward': [],
        'std_reward': []
    }
    
    # Grid search
    for lr, ts, gamma, clip, n_steps in product(lr_values, ts_values, gamma_values, clip_values, n_steps_values):
        # Train model with current parameters
        np.random.seed(0)
        env = Monitor(FlexSimEnv(reward_weights={'off_schedule_trips': -2.0, 'lost_requests': -1.0}))
        model = PPO("MultiInputPolicy", env, verbose=verbose, 
                   n_steps=n_steps, 
                   learning_rate=lr,
                   gamma=gamma, 
                   clip_range=clip)
        
        model.learn(total_timesteps=int(ts))
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
        
        ## Store in results
        results['lr'].append(lr)
        results['ts'].append(ts)
        results['gamma'].append(gamma)
        results['clip'].append(clip)
        results['mean_reward'].append(mean_reward)
        results['std_reward'].append(std_reward)
        results['n_steps'].append(n_steps)       
 
        print(f"Params: lr={lr}, ts={ts}, gamma={gamma}, clip={clip}, n_steps={n_steps}")
        print(f"Reward: {round(mean_reward, 3)} +/- {round(std_reward, 3)}")
        print("------------------------")
        
        env.close()
    
    return results

# Run the training with the default parameters
# train_flexsim(save=True, test=True)

# Run the training with the default parameters or pass specific values for testing
# train_flexsim(n_steps=256, total_timesteps=16000, verbose=0, save=False, test=True, learning_rate=0.0005, gamma=0.99, clip_range=0.2)

