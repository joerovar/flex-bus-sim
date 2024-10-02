from objects import RouteManager, EventManager
from helpers import *
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
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

    # def step(self, action):
    #     ...
    #     return observation, reward, terminated, truncated, info

    # def reset(self, seed=None, options=None):
    #     ...
    #     return observation, info

    # def render(self):
    #     ...

    # def close(self):
    #     ...


## Experimental Design Parameters
STEPS_PER_EPISODE = 50
N_EPISODES = 10

pax_results = []
trip_results = []

if __name__ == '__main__':
    ## evaluation
    ## SCENARIO DO NOTHING
    for i in range(N_EPISODES):
        route = RouteManager()
        event = EventManager()
        event.start_vehicles(route)
        route.load_all_pax()

        steps = 0
        obs, reward, terminated, truncated, info = event.step(route, action=True)
        while not terminated:
            obs, reward, terminated, truncated, info = event.step(route, action=True)
            print(f'Time {event.timestamps[-1]}')
            print(info)
            steps += 1


    ## SCENARIO SERVE EVERYONE

    # vehicles, drivers = setup_simulation(env, routes, stops)
    # env.run(until=20)
