import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import EnvironmentManager

class FlexSimEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, reward_weight=TRIP_WEIGHT):
        self.observation_space = spaces.Dict({
            "control_stop_idx": spaces.Discrete(4),  # there are four control stops in the route
            "n_requests": spaces.Discrete(5),  # Clip any values above 4
            "headway": spaces.Box(0.0, 1400.0, (1,), dtype=np.float32),  # headway (continuous)
            "schedule_deviation": spaces.Box(-1200.0, 1200.0, (1,), dtype=np.float32),  # delay (continuous)
        })

        ## Action space remains discrete (binary actions: deviate or not)
        self.action_space = spaces.Discrete(2)

        # Initialize other environment parameters
        self.env = None
        self.route = None

        # Initialize reward weights
        self.reward_weight = reward_weight

    def get_obs_dict(self, observation):
        control_stop_index = observation[0]
        n_requests = np.int32(min(observation[1], np.int32(4)))
        headway = observation[2]
        schedule_deviation = observation[3]
        
        # Make sure the observation is returned as a dictionary matching the observation space
        obs_dict = {
            "control_stop_idx": np.array([control_stop_index], dtype=np.int32),
            "n_requests": np.array([n_requests], dtype=np.int32),
            "headway": np.array([headway], dtype=np.float32),
            "schedule_deviation": np.array([schedule_deviation], dtype=np.float32),
        }
        return obs_dict

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self.get_obs_dict(observation)
        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.env = EnvironmentManager(reward_weight=self.reward_weight)
        self.env.start_vehicles()
        self.env.route.load_all_pax()
        observation, _, _, _, info = self.env.step(action=None)

        obs_dict = self.get_obs_dict(observation)
        return obs_dict, info

    def render(self):
        pass

    def close(self):
        pass




