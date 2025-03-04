from objects import EnvironmentManager
from helpers import *
import os
from datetime import datetime
from rl_env import *
from stable_baselines3 import PPO

## Experimental Design Parameters
N_EPISODES = 25

results = {'pax': [], 'vehicles': [], 'state': [], 'idle': []}

BASE_SCENARIOS = ['RA', 'ND']

if __name__ == '__main__':
    ## evaluation
    # base scenarios
    for scenario in BASE_SCENARIOS:
        ## set seeds
        np.random.seed(0)

        for i in range(N_EPISODES):
            # route = RouteManager()
            env = EnvironmentManager()
            env.start_vehicles()
            env.route.load_all_pax()

            obs, reward, terminated, truncated, info = env.step(action=None)
            while not terminated:
                action = get_action(scenario, obs)
                obs, reward, terminated, truncated, info = env.step(action=action)
    
            history = env.get_history()
            for key in history:
                history[key]['scenario'] = scenario
                history[key]['episode'] = i
                results[key].append(history[key])

    # dynamic rule deviation
    min_pax_per_sched_dev = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    for min_pax in min_pax_per_sched_dev:
        np.random.seed(0)
        for i in range(N_EPISODES):
            # route = RouteManager()
            env = EnvironmentManager()
            env.start_vehicles()
            env.route.load_all_pax()

            obs, reward, terminated, truncated, info = env.step(action=None)
            while not terminated:
                action = get_action('DRD', obs, min_pax)
                obs, reward, terminated, truncated, info = env.step(action=action)
    
            history = env.get_history()
            for key in history:
                history[key]['scenario'] = 'DRD_' + str(int(min_pax*10))
                history[key]['episode'] = i
                results[key].append(history[key])

    # RL agent
    models = ['ppo_10', 'ppo_20', 'ppo_30']
    for model_name in models:
        np.random.seed(0)
        model = PPO.load(f'models/{model_name}')
        for i in range(N_EPISODES):
            rl_env = FlexSimEnv()

            obs, info = rl_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            action = int(action[0])
            obs, reward, terminated, truncated, info = rl_env.step(action=action)
            while not terminated:
                # Use the loaded agent to predict the action based on current observation
                action, _ = model.predict(obs, deterministic=True)
                action = int(action[0])
                obs, reward, terminated, truncated, info = rl_env.step(action=action)

            history = rl_env.env.get_history()
            for key in history:
                history[key]['scenario'] = 'RL_' + model_name
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