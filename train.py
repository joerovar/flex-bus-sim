import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import *

## STATE PARAMETERS
## [stop_idx, n_flex_pax, headway, load, delay]
## stop_idx: index of the stop in the route
## n_flex_pax: number of flexible passengers at the stop
## headway: time between the current and last vehicle
## load: number of passengers on the vehicle
## delay: difference between the scheduled and actual arrival time

class IndependentAgents(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=np.array(LOW), high=np.array(HIGH),
                                            shape=(STATE_DIM, ), dtype=np.uint8)
        
        self.env = None

    def step(self, action):
        ...
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        route = RouteManager()
        event = EventManager()
        event.start_vehicles(route)
        route.load_all_pax()
        observation, _, _, info = event.step(route, action=None)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass