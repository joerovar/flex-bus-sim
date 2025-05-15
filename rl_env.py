import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import EnvironmentManager

class FlexSimEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    def __init__(self, reward_weight=TRIP_WEIGHT, demand_scenario='peak'):
        self.observation_space = spaces.Dict({
            "control_stop_idx": spaces.Discrete(4),  # there are four control stops in the route
            "n_requests": spaces.Discrete(5),  # Clip any values above 4
            "headway": spaces.Discrete(141),  # headway in 10s intervals (0-1400s → 0-140 integers)
            "schedule_deviation": spaces.Box(-120, 120, (1,), dtype=np.int32),  # delay in 10s intervals (-1200s to +1200s → -120 to +120 integers)
        })

        ## Action space remains discrete (binary actions: deviate or not)
        self.action_space = spaces.Discrete(2)

        # Initialize other environment parameters
        self.env = None
        self.route = None

        # Initialize reward weights
        self.reward_weight = reward_weight

        # demand scenario
        self.demand_scenario = demand_scenario

        # experience log
        self.experience_log = {
            'veh_idx': [],
            'time': [],
            'action': [],
            'reward': [],
            'done': [],
            'next_obs': [],
        }

    def get_obs_dict(self, observation):
        control_stop_index = observation[0]
        n_requests = np.int32(min(observation[1], np.int32(4)))
        
        # Convert continuous headway to 10s resolution integer
        headway_int = np.int32(round(observation[2] / 10.0))
        
        # Convert continuous schedule deviation to 10s resolution integer
        schedule_deviation_int = np.int32(round(observation[3] / 10.0))
        
        # Make sure the observation is returned as a dictionary matching the observation space
        obs_dict = {
            "control_stop_idx": np.array([control_stop_index], dtype=np.int32),
            "n_requests": np.array([n_requests], dtype=np.int32),
            "headway": np.array([headway_int], dtype=np.int32),
            "schedule_deviation": np.array([schedule_deviation_int], dtype=np.int32),
        }
        return obs_dict

    def step(self, action):
        observation, reward, done, terminated, info = self.env.step(action)
        obs_dict = self.get_obs_dict(observation)

        # Log the experience
        self.experience_log['time'].append(info['time'])
        self.experience_log['next_obs'].append(observation)
        self.experience_log['action'].append(action)
        self.experience_log['reward'].append(reward)
        self.experience_log['done'].append(done)
        self.experience_log['veh_idx'].append(info['veh_idx'])

        return obs_dict, reward, done, terminated, info

    def reset(self, seed=None, options=None):
        self.env = EnvironmentManager(reward_weight=self.reward_weight)
        self.env.start_vehicles()
        self.env.route.load_all_pax(demand_scenario=self.demand_scenario)
        observation, _, _, _, info = self.env.step(action=None)

        obs_dict = self.get_obs_dict(observation)
        # reset the experience log
        self.experience_log = {
            'veh_idx': [],
            'time': [],
            'action': [],
            'reward': [],
            'done': [],
            'next_obs': [],
        }
        return obs_dict, info

    def render(self):
        pass

    def close(self):
        pass




