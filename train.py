import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import *

STATE_KEYS = ['stop_idx', 'n_flex_pax', 'load', 'headway', 'delay']

class IndependentAgents(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        ## Define the mixed observation space components

        # stop_idx: index of the stop in the route (discrete space, 7 stops: 0-6)
        stop_idx_space = spaces.Discrete(7)

        # n_flex_pax: discrete number of flexible passengers (max 7 passengers)
        n_flex_pax_space = spaces.Discrete(8)  # 0 to 7 passengers

        # headway: continuous time between the current and last vehicle (e.g., [0, 20] minutes)
        headway_space = spaces.Box(low=0, high=20, shape=(1,), dtype=np.float32)

        # load: discrete load (number of passengers on the vehicle, max 20 passengers)
        load_space = spaces.Discrete(21)  # 0 to 20 passengers

        # delay: continuous delay (difference between scheduled and actual arrival in minutes, e.g., [-10, 10])
        delay_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        ## Combine the observation spaces using a Tuple
        self.observation_space = spaces.Tuple((
            stop_idx_space,    # Discrete integer for stop index
            n_flex_pax_space,  # Discrete integer for number of flexible passengers
            headway_space,     # Continuous float for headway
            load_space,        # Discrete integer for load
            delay_space        # Continuous float for delay
        ))

        ## Action space remains discrete (binary actions: deviate or not)
        self.action_space = spaces.Discrete(2)

        # Initialize other environment parameters
        self.env = None
        self.route = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(self.route, action)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.route = RouteManager()
        self.env = EventManager()
        self.env.start_vehicles(self.route)
        self.route.load_all_pax()
        observation, _, _, _, info = self.env.step(self.route, action=None)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass