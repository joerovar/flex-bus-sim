from objects import EnvironmentManager
from helpers import *
import os
from datetime import datetime
from train import *

## Experimental Design Parameters
N_EPISODES = 30

pax_results = []
trip_results = []
state_results = []
idle_results = []

SCENARIOS = ['ND', 'AD', 'RA', 'FRD', 'DRD']

if __name__ == '__main__':
    ## evaluation
    
    ## SCENARIO NEVER DEVIATE (ND)
    for scenario in SCENARIOS:
        ## set seeds
        np.random.seed(0)

        for i in range(N_EPISODES):
            # route = RouteManager()
            env = EnvironmentManager()
            env.start_vehicles()
            env.route.load_all_pax()

            obs, reward, terminated, truncated, info = env.step(action=None)
            while not terminated:
                action = get_action(scenario, obs, min_pax_thresholds=DEFAULT_MIN_PAX_THRESHOLDS)
                obs, reward, terminated, truncated, info = env.step(action=action)
    
            pax_hist = get_pax_hist(env.route, FLEX_STOPS, include_denied=True)
            veh_hist = get_vehicle_history(env.route.vehicles, FLEX_STOPS)
            state_hist = pd.DataFrame(env.state_hist)
            idle_hist = pd.DataFrame(env.route.idle_time)

            for hist in (pax_hist, veh_hist, state_hist, idle_hist):
                hist['scenario'] = scenario
                hist['episode'] = i

            pax_results.append(pax_hist)
            trip_results.append(veh_hist)
            state_results.append(state_hist)
            idle_results.append(idle_hist)

    scenario = 'RL'
    model = PPO.load("ppo_flexsim")

    rewards = []
    for i in range(N_EPISODES):
        rl_env = FlexSimEnv()

        obs, info = rl_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = rl_env.step(action=action)
        while not terminated:
            # Use the loaded agent to predict the action based on current observation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = rl_env.step(action=action)

        pax_hist = get_pax_hist(rl_env.env.route, FLEX_STOPS, include_denied=True)
        veh_hist = get_vehicle_history(rl_env.env.route.vehicles, FLEX_STOPS)
        state_hist = pd.DataFrame(rl_env.env.state_hist)
        idle_hist = pd.DataFrame(rl_env.env.route.idle_time)

        for hist in (pax_hist, veh_hist, state_hist, idle_hist):
            hist['scenario'] = scenario
            hist['episode'] = i

        pax_results.append(pax_hist)
        trip_results.append(veh_hist)
        state_results.append(state_hist)
        idle_results.append(idle_hist)

# Get the current date and time
now = datetime.now()

# Format the folder name as 'experiments_MMDD-HHMMSS'
folder_name = now.strftime("experiments_%m%d-%H%M%S")

# Define the path where you want to create the folder
folder_path = os.path.join(OUTPUT_FOLDER_PATH, folder_name)

# Create the folder
os.mkdir(folder_path)

print(f"Folder created: {folder_path}")

# Save the concatenated DataFrames to CSV files using os.path.join for the file paths
pax_results_df = pd.concat(pax_results)
pax_results_df.to_csv(os.path.join(folder_path, 'pax.csv'), index=False)

trip_results_df = pd.concat(trip_results)
trip_results_df.to_csv(os.path.join(folder_path, 'trips.csv'), index=False)

state_results_df = pd.concat(state_results)
state_results_df.to_csv(os.path.join(folder_path, 'state.csv'), index=False)

idle_results_df = pd.concat(idle_results)
idle_results_df.to_csv(os.path.join(folder_path, 'idle.csv'), index=False)