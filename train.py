import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import EnvironmentManager
## import the PPO agent from stable_baselines
from stable_baselines3 import PPO
## import the evaluate_policy function
from stable_baselines3.common.evaluation import evaluate_policy

STATE_KEYS = ['stop_idx', 'n_flex_pax', 'load', 'headway', 'delay']

class FlexSimEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self):
        self.observation_space = spaces.Dict({
            "stop_idx": spaces.Box(0.0, 6.0, (1,), dtype=np.float32),    # Replaces Discrete(7)
            "n_flex_pax": spaces.Box(0.0, 7.0, (1,), dtype=np.float32),    # Replaces Discrete(8)
            "headway": spaces.Box(0.0, 1200.0, (1,), dtype=np.float32),  # headway (continuous)
            "load": spaces.Box(0.0, 29.0, (1,), dtype=np.float32),   # Replaces Discrete(30)
            "delay": spaces.Box(-600.0, 600.0, (1,), dtype=np.float32) # delay (continuous)
        })

        ## Action space remains discrete (binary actions: deviate or not)
        self.action_space = spaces.Discrete(2)

        # Initialize other environment parameters
        self.env = None
        self.route = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Make sure the observation is returned as a dictionary matching the observation space
        obs_dict = {
            "stop_idx": np.array([observation[0]], dtype=np.float32),
            "n_flex_pax": np.array([observation[1]], dtype=np.float32),
            "headway": np.array([observation[2]], dtype=np.float32),
            "load": np.array([observation[3]], dtype=np.float32),
            "delay": np.array([observation[4]], dtype=np.float32)
        }

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env = EnvironmentManager()
        self.env.start_vehicles()
        self.env.route.load_all_pax()
        observation, _, _, _, info = self.env.step(action=None)

        # Ensure the initial observation is a dictionary
        obs_dict = {
            "stop_idx": np.array([observation[0]], dtype=np.float32),
            "n_flex_pax": np.array([observation[1]], dtype=np.float32),
            "headway": np.array([observation[2]], dtype=np.float32),
            "load": np.array([observation[3]], dtype=np.float32),
            "delay": np.array([observation[4]], dtype=np.float32)
        }

        return obs_dict, info

    def render(self):
        pass

    def close(self):
        pass

from stable_baselines3.common.monitor import Monitor

# function to train a simple PPO agent on the FlexSim environment
def train_flexsim(n_steps=200, total_timesteps=1200, 
                  verbose=1, save=False, test=False, learning_rate=0.0003, gamma=0.99, clip_range=0.2):
    # env = Monitor(FlexSimEnv())
    env = FlexSimEnv()
    env.reset()

    # Initialize the PPO agent with specified n_steps and verbosity
    model = PPO("MultiInputPolicy", env, verbose=verbose, n_steps=n_steps, learning_rate=learning_rate,
                gamma=gamma, clip_range=clip_range)
    
    # Train the agent for the specified number of timesteps
    model.learn(total_timesteps=total_timesteps)

    # Save the agent if save=True
    if save:
        model.save("ppo_flexsim")
        print("Agent saved as 'ppo_flexsim'.")

    if test:
        model = PPO.load("ppo_flexsim")
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        print('hyperpatemers')
        print(learning_rate, gamma, clip_range)
        print('------')

# Run the training with the default parameters or pass specific values for testing
# train_flexsim(n_steps=256, total_timesteps=16000, verbose=0, save=True, test=True, learning_rate=0.00031, gamma=0.99, clip_range=0.2)
# train_flexsim(n_steps=256, total_timesteps=16000, verbose=0, save=False, test=True, learning_rate=0.00035, gamma=0.99, clip_range=0.2)
# train_flexsim(n_steps=256, total_timesteps=15000, verbose=0, save=False, test=True, learning_rate=0.0007, gamma=0.98, clip_range=0.15)