from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from objects import EnvironmentManager
from helpers import *

def run_loaded_agent(N_EPISODES=10, scenario='scenario1'):
    # Load the trained agent
    agent = PPO.load("ppo_flexsim")
    print("Agent loaded from 'ppo_flexsim'.")

    # Initialize lists to store history data
    all_pax_hist = []
    all_veh_hist = []
    all_state_hist = []
    all_idle_hist = []

    for i in range(N_EPISODES):
        np.random.seed(0)
        env = EnvironmentManager()
        env.start_vehicles()
        env.route.load_all_pax()

        obs, reward, terminated, truncated, info = env.step(action=None)

        while not terminated:
            # Use the loaded agent to predict the action based on current observation
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action=action)

        # # Collect episode history
        # pax_hist = get_pax_hist(env.route, FLEX_STOPS, include_denied=True)
        # veh_hist = get_vehicle_history(env.route.vehicles, FLEX_STOPS)
        state_hist = pd.DataFrame(env.state_hist)
        # idle_hist = pd.DataFrame(env.route.idle_time)

        # # Add scenario and episode info to each DataFrame
        # for hist in (pax_hist, veh_hist, state_hist, idle_hist):
        #     hist['scenario'] = scenario
        #     hist['episode'] = i

        # # Append histories to the lists
        # all_pax_hist.append(pax_hist)
        # all_veh_hist.append(veh_hist)
        all_state_hist.append(state_hist)
        # all_idle_hist.append(idle_hist)


    print(all_state_hist['reward'].mean())
    # Concatenate all histories into final DataFrames
    # final_pax_hist = pd.concat(all_pax_hist, ignore_index=True)
    # final_veh_hist = pd.concat(all_veh_hist, ignore_index=True)
    # final_state_hist = pd.concat(all_state_hist, ignore_index=True)
    # final_idle_hist = pd.concat(all_idle_hist, ignore_index=True)
    
    # return final_pax_hist, final_veh_hist, final_state_hist, final_idle_hist