import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import EnvironmentManager


STATE_KEYS = ['stop_idx', 'n_flex_pax', 'load', 'headway', 'delay', 'prev_action']

class FlexSimEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, reward_weights=REWARD_WEIGHTS):
        self.observation_space = spaces.Dict({
            "stop_idx": spaces.Discrete(2),  # 0 or 1 (converted from CONTROL_STOPS [1,3])
            "n_flex_pax": spaces.Discrete(8),  # Clip any values above 7
            "headway": spaces.Box(0.0, 1200.0, (1,), dtype=np.float32),  # headway (continuous)
            "load": spaces.Box(0.0, 29.0, (1,), dtype=np.float32),  # load (continuous)
            "delay": spaces.Box(-600.0, 600.0, (1,), dtype=np.float32),  # delay (continuous)
            "prev_action": spaces.Discrete(2)  # Previous action: 0 or 1
        })

        ## Action space remains discrete (binary actions: deviate or not)
        self.action_space = spaces.Discrete(2)

        # Initialize other environment parameters
        self.env = None
        self.route = None

        # Initialize reward weights
        self.reward_weights = reward_weights

    def get_obs_dict(self, observation):
        # Convert `stop_idx` based on CONTROL_STOPS and clip `n_flex_pax`
        stop_idx = 0 if observation[0] == CONTROL_STOPS[0] else 1
        n_flex_pax = min(observation[1], 7)
        
        # Make sure the observation is returned as a dictionary matching the observation space
        obs_dict = {
            "stop_idx": np.array([stop_idx], dtype=np.int32),
            "n_flex_pax": np.array([n_flex_pax], dtype=np.int32),
            "headway": np.array([observation[2]], dtype=np.float32),
            "load": np.array([observation[3]], dtype=np.float32),
            "delay": np.array([observation[4]], dtype=np.float32),
            "prev_action": np.array([observation[5]], dtype=np.int32)  # previous action of prior vehicle
        }
        return obs_dict

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self.get_obs_dict(observation)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env = EnvironmentManager(reward_weights=self.reward_weights)
        self.env.start_vehicles()
        self.env.route.load_all_pax()
        observation, _, _, _, info = self.env.step(action=None)

        obs_dict = self.get_obs_dict(observation)
        return obs_dict, info

    def render(self):
        pass

    def close(self):
        pass




