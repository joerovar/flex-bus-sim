from objects import EnvironmentManager
from helpers import *
import os
from datetime import datetime

## Experimental Design Parameters
N_EPISODES = 30

pax_results = []
trip_results = []
state_results = []
idle_results = []

SCENARIO = 'DRD'

PARAMETERS = [
[(0, 1), (120, 2), (240, 4) ,(1000, 5)], ## medium
[(0, 1), (90, 2), (180, 4) ,(1000, 5)], ## more strict
[(0, 1), (60, 2), (180, 4) ,(1000, 5)], ## more strict
[(0, 1), (60, 2), (120, 4) ,(1000, 5)], ## more strict
[(0, 1), (30, 2), (120, 4) ,(1000, 5)] ## more strict
]

if __name__ == '__main__':
    ## evaluation
    
    for j, parameter in enumerate(PARAMETERS):
        ## set seeds
        np.random.seed(0)

        for i in range(N_EPISODES):
            # route = RouteManager()
            env = EnvironmentManager()
            env.start_vehicles()
            env.route.load_all_pax()

            obs, reward, terminated, truncated, info = env.step(action=None)
            while not terminated:
                action = get_action(SCENARIO, obs, min_pax_thresholds=parameter)
                obs, reward, terminated, truncated, info = env.step(action=action)
    
            pax_hist = get_pax_hist(env.route, FLEX_STOPS, include_denied=True)
            veh_hist = get_vehicle_history(env.route.vehicles, FLEX_STOPS)
            state_hist = pd.DataFrame(env.state_hist)
            idle_hist = pd.DataFrame(env.route.idle_time)

            for hist in (pax_hist, veh_hist, state_hist, idle_hist):
                hist['scenario'] = SCENARIO + '_param_' + str(j)
                hist['episode'] = i

            pax_results.append(pax_hist)
            trip_results.append(veh_hist)
            state_results.append(state_hist)
            idle_results.append(idle_hist)

# Get the current date and time
now = datetime.now()

# Format the folder name as 'experiments_MMDD-HHMMSS'
folder_name = now.strftime(f"{SCENARIO}_calibration_experiments_%m%d-%H%M%S")

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