from objects import EnvironmentManager
from helpers import *
import os
from datetime import datetime
from rl_env import *

## Experimental Design Parameters
N_EPISODES = 25

results = {'pax': [], 'vehicles': [], 'state': [], 'idle': []}

if __name__ == '__main__':
    ## evaluation
    # dynamic rule deviation
    base_minimum_requests = BASE_MINIMUM_REQUEST
    minimum_request_slopes = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0] 
    for minimum_request_slope in minimum_request_slopes:
        np.random.seed(0)
        for i in range(N_EPISODES):
            env = EnvironmentManager()
            env.start_vehicles()
            env.route.load_all_pax()

            observation, reward, terminated, truncated, info = env.step(action=None)
            while not terminated:
                action = get_action('DRD', observation, minimum_request_slope)
                observation, reward, terminated, truncated, info = env.step(action=action)
    
            history = env.get_history()
            for key in history:
                history[key]['scenario'] = 'DRD_' + str(int(minimum_request_slope*10))
                history[key]['episode'] = i
                results[key].append(history[key])

    # Get the current date and time
    now = datetime.now()

    # Format the folder name as 'experiments_MMDD-HHMMSS'
    folder_name = now.strftime("experiments_%m%d-%H%M%S")

    # Define the path where you want to create the folder
    folder_path = os.path.join(OUTPUT_FOLDER_PATH, folder_name)

    # Create the folder
    os.mkdir(folder_path)

    print(f"Folder created: {folder_path}")
    
    # For debugging, save results
    # Save the concatenated DataFrames to CSV files using os.path.join for the file paths
    for df in results:
        results[df] = pd.concat(results[df])
        results[df].to_csv(os.path.join(folder_path, f'{df}.csv'), index=False)