from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from rl_env import FlexSimEnv
import pandas as pd
from itertools import product
import numpy as np

# function to train a simple PPO agent on the FlexSim environment
def train_flexsim(n_steps=64, total_timesteps=1200, 
                  verbose=0, save=False, test=False, learning_rate=0.0003, gamma=0.99, clip_range=0.2):
    env = Monitor(FlexSimEnv())
    # env = FlexSimEnv()
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
        print('hyperpatemers')
        print(learning_rate, gamma, clip_range)
        print('------')

    # Save the agent if save=True
    if save:
        model.save("models/ppo_flexsim")
        print("Agent saved as 'ppo_flexsim'.")

def grid_search_flexsim(
    learning_rate_range=(0.0003, 0.0006, 0.0001),
    total_timesteps_range=(12000, 16000, 2000),
    gamma_range=(0.98, 0.99, 0.01),
    clip_range_range=(0.15, 0.2, 0.05),
    n_steps=256,
    verbose=0
):
    # Generate parameter combinations
    lr_values = np.arange(*learning_rate_range)
    ts_values = np.arange(*total_timesteps_range)
    gamma_values = np.arange(*gamma_range)
    clip_values = np.arange(*clip_range_range)
    
    # Dictionary to store results
    results = {
        'lr': [],
        'ts': [],
        'gamma': [],
        'clip': [],
        'mean_reward': [],
        'std_reward': []
    }
    best_reward = float('-inf')
    
    # Grid search
    for lr, ts, gamma, clip in product(lr_values, ts_values, gamma_values, clip_values):
        # Train model with current parameters
        env = Monitor(FlexSimEnv())
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
        
        # Update best model if current is better
        if mean_reward > best_reward:
            best_reward = mean_reward
            # Save current best model
            model.save("models/ppo_flexsim")
            
        print(f"Params: lr={lr}, ts={ts}, gamma={gamma}, clip={clip}")
        print(f"Reward: {round(mean_reward, 3)} +/- {round(std_reward, 3)}")
        print("------------------------")
        
        env.close()
    
    return results

# Example usage:
results, best_params, best_reward = grid_search_flexsim(
    learning_rate_range=(0.0003, 0.0007, 0.0001),
    total_timesteps_range=(14000, 18000, 2000),
    gamma_range=(0.98, 1.0, 0.01),
    clip_range_range=(0.15, 0.25, 0.05)
)

df_results = pd.DataFrame(results)
df_results.to_csv('outputs/grid_search_results.csv', index=False)

print("\nBest parameters:", best_params)
print("Best reward:", best_reward)

# Run the training with the default parameters
# train_flexsim(save=True, test=True)

# Run the training with the default parameters or pass specific values for testing
# train_flexsim(n_steps=256, total_timesteps=16000, verbose=0, save=False, test=True, learning_rate=0.00031, gamma=0.99, clip_range=0.2)