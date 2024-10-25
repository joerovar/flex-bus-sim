import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import *
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
        observation, reward, terminated, truncated, info = self.env.step(self.route, action)
        if len(observation) == 0:
            print(observation, reward, terminated, truncated)
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
        self.route = RouteManager()
        self.env = EventManager()
        self.env.start_vehicles(self.route)
        self.route.load_all_pax()
        observation, _, _, _, info = self.env.step(self.route, action=None)

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

# function to train a simple PPO agent on the FlexSim environment
def train_flexsim():
    env = FlexSimEnv()
    # env.seed(0)
    env.reset()

    # Train the agent
    agent = PPO("MultiInputPolicy", env, verbose=1)
    agent.learn(total_timesteps=100)

    # Save the trained agent
    agent.save("ppo_flexsim")

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Close the environment
    env.close()

train_flexsim()