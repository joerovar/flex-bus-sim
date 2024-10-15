import gymnasium as gym
import numpy as np
from gymnasium import spaces
from params import *
from objects import *

## STATE PARAMETERS
## [stop_idx, n_flex_pax, headway, load, delay]
## stop_idx: index of the stop in the route (7 bins: 0-6)
## n_flex_pax: number of flexible passengers at the stop (3 bins: 0, 1, 2+)
## headway: time between the current and last vehicle (3 bins: 0-5, 5-10, 10+)
## load: number of passengers on the vehicle (2 bins: 0-2, 3+)
## delay: difference between the scheduled and actual arrival time (3 bins: <1, 1-3, 3+)




def get_bin_index(state_name, actual_value):
    """
    Maps an actual value for a state variable to its corresponding bin index.

    Parameters:
    - state_name: The name of the state variable (e.g., 'n_flex_pax', 'headway', 'load', 'delay')
    - actual_value: The actual value of the state variable

    Returns:
    - The index of the bin that the actual value falls into
    """
    bounds = PARAM_BOUNDS[state_name]

    # If bounds is a list, it's directly indexed (for stop_idx)
    if isinstance(bounds, list):
        return min(max(0, actual_value), bounds[1])

    # For others, the bounds are defined as bins with maximum values
    for i, bound in enumerate(bounds['bins']):
        if actual_value < bound:
            return i

    # If the value exceeds all bounds, return the last bin
    return len(bounds['bins'])

# Example usage:
# To get the bin index for 'headway' with an actual value of 7:
# bin_index = get_bin_index('headway', 7)   # Returns 1, because 7 falls into the 5-10 range

class IndependentAgents(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        ## Define the discrete observation space components
        # stop_idx: 7 possible stops (0 to 6)
        stop_idx_space = spaces.Discrete(7)

        # n_flex_pax: number of flexible passengers (3 bins: 0, 1, 2+)
        n_flex_pax_space = spaces.Discrete(3)

        # headway: time between vehicles (3 bins: 0-5, 5-10, 10+)
        headway_space = spaces.Discrete(3)

        # load: number of passengers on the vehicle (2 bins: 0-2, 3+)
        load_space = spaces.Discrete(2)

        # delay: difference between scheduled and actual arrival (3 bins: <1, 1-3, 3+)
        delay_space = spaces.Discrete(3)

        ## Combine them into a tuple of observation spaces
        self.observation_space = spaces.Tuple((
            stop_idx_space,
            n_flex_pax_space,
            headway_space,
            load_space,
            delay_space
        ))

        # Define the action space: binary (deviate or not)
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
        observation, _, _, info = self.env.step(self.route, action=None)
        return observation, info

    def render(self):
        pass

    def close(self):
        pass