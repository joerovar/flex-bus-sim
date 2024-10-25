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

##TODO: Modularize the code to allow for different scenarios
##TODO: Add seed and verify reproducibility

if __name__ == '__main__':
    ## evaluation
    
    ## SCENARIO NEVER DEVIATE (ND)
    SCENARIO = 'ND'

    for i in range(N_EPISODES):
        # route = RouteManager()
        env = EnvironmentManager()
        env.start_vehicles()
        env.route.load_all_pax()

        obs, reward, terminated, truncated, info = env.step(action=None)
        while not terminated:
            action = get_action(SCENARIO, obs)
            obs, reward, terminated, truncated, info = env.step(action=action) ## action 0 is to never deviate
 
        # pax_hist = get_pax_hist(route, FLEX_STOPS, include_denied=True)
        # veh_hist = get_vehicle_history(route.vehicles, FLEX_STOPS)
        # state_hist = pd.DataFrame(event.state_hist)
        # idle_hist = pd.DataFrame(route.idle_time)

        # for hist in (pax_hist, veh_hist, state_hist, idle_hist):
        #     hist['scenario'] = SCENARIO
        #     hist['episode'] = i

        # pax_results.append(pax_hist)
        # trip_results.append(veh_hist)
        # state_results.append(state_hist)
        # idle_results.append(idle_hist)

#     ## SCENARIO ALWAYS DEVIATE
#     for i in range(N_EPISODES):
#         SCENARIO = 'AD'
#         route = RouteManager()
#         event = EventManager()
#         event.start_vehicles(route)
#         route.load_all_pax()

#         obs, reward, terminated, truncated, info = event.step(route, action=None)
#         while not terminated:
#             n_pax = obs[1]
#             if n_pax:
#                 obs, reward, terminated, truncated, info = event.step(route, action=1) ## deviate   
#             else:
#                 obs, reward, terminated, truncated, info = event.step(route, action=0) ## do not deviate  

#         pax_hist = get_pax_hist(route, FLEX_STOPS, include_denied=True)
#         veh_hist = get_vehicle_history(route.vehicles, FLEX_STOPS)
#         state_hist = pd.DataFrame(event.state_hist)
#         idle_hist = pd.DataFrame(route.idle_time)

#         for hist in (pax_hist, veh_hist, state_hist, idle_hist):
#             hist['scenario'] = SCENARIO
#             hist['episode'] = i

#         pax_results.append(pax_hist)
#         trip_results.append(veh_hist)
#         state_results.append(state_hist)
#         idle_results.append(idle_hist)

#     ## SCENARIO RANDOM ACTION
#     for i in range(N_EPISODES):
#         SCENARIO = 'RA'
#         route = RouteManager()
#         event = EventManager()
#         event.start_vehicles(route)
#         route.load_all_pax()

#         obs, reward, terminated, truncated, info = event.step(route, action=None)
#         while not terminated:
#             n_pax = obs[1]
#             action = np.random.choice([0,1])
#             obs, reward, terminated, truncated, info = event.step(route, action=action)   

#         pax_hist = get_pax_hist(route, FLEX_STOPS, include_denied=True)
#         veh_hist = get_vehicle_history(route.vehicles, FLEX_STOPS)
#         state_hist = pd.DataFrame(event.state_hist)
#         idle_hist = pd.DataFrame(route.idle_time)

#         for hist in (pax_hist, veh_hist, state_hist, idle_hist):
#             hist['scenario'] = SCENARIO
#             hist['episode'] = i

#         pax_results.append(pax_hist)
#         trip_results.append(veh_hist)
#         state_results.append(state_hist)
#         idle_results.append(idle_hist)

#     ## SCENARIO SELECTIVE DEVIATION: SERVE ONLY IF THE SCHEDULE DELAY UPON DEPARTURE IS WIHTIN A THRESHOLD
#     for i in range(N_EPISODES):
#         SCENARIO = 'SD'
#         route = RouteManager()
#         event = EventManager()
#         event.start_vehicles(route)
#         route.load_all_pax()

#         obs, reward, terminated, truncated, info = event.step(route, action=None)
#         while not terminated:
#             delay = obs[4]

#             if delay < DELAY_THRESHOLD:
#                 obs, reward, terminated, truncated, info = event.step(route, action=1)   
#             else:
#                 obs, reward, terminated, truncated, info = event.step(route, action=0)  

#         pax_hist = get_pax_hist(route, FLEX_STOPS, include_denied=True)
#         veh_hist = get_vehicle_history(route.vehicles, FLEX_STOPS)
#         state_hist = pd.DataFrame(event.state_hist)
#         idle_hist = pd.DataFrame(route.idle_time)

#         for hist in (pax_hist, veh_hist, state_hist, idle_hist):
#             hist['scenario'] = SCENARIO
#             hist['episode'] = i

#         pax_results.append(pax_hist)
#         trip_results.append(veh_hist)
#         state_results.append(state_hist)
#         idle_results.append(idle_hist)
    
#     ## SCENARIO DYNAMIC SELECTIVE DEVIATION: SERVE ONLY IF THE SCHEDULE DELAY UPON DEPARTURE IS WIHTIN A THRESHOLD
#     for i in range(N_EPISODES):
#         SCENARIO = 'DSD'
#         route = RouteManager()
#         event = EventManager()
#         event.start_vehicles(route)
#         route.load_all_pax()

#         obs, reward, terminated, truncated, info = event.step(route, action=None)
#         while not terminated:
#             n_pax = obs[1]
#             delay = obs[4]
            
#             min_pax = get_min_pax_threshold(delay)
#             if n_pax >= min_pax:
#                 obs, reward, terminated, truncated, info = event.step(route, action=1)
#             else:
#                 obs, reward, terminated, truncated, info = event.step(route, action=0)

#         pax_hist = get_pax_hist(route, FLEX_STOPS, include_denied=True)
#         veh_hist = get_vehicle_history(route.vehicles, FLEX_STOPS)
#         state_hist = pd.DataFrame(event.state_hist)
#         idle_hist = pd.DataFrame(route.idle_time)

#         for hist in (pax_hist, veh_hist, state_hist, idle_hist):
#             hist['scenario'] = SCENARIO
#             hist['episode'] = i

#         pax_results.append(pax_hist)
#         trip_results.append(veh_hist)
#         state_results.append(state_hist)
#         idle_results.append(idle_hist)


# # Get the current date and time
# now = datetime.now()

# # Format the folder name as 'experiments_MMDD-HHMMSS'
# folder_name = now.strftime("experiments_%m%d-%H%M%S")

# # Define the path where you want to create the folder
# folder_path = os.path.join(OUTPUT_FOLDER_PATH, folder_name)

# # Create the folder
# os.mkdir(folder_path)

# print(f"Folder created: {folder_path}")

# # Save the concatenated DataFrames to CSV files using os.path.join for the file paths
# pax_results_df = pd.concat(pax_results)
# pax_results_df.to_csv(os.path.join(folder_path, 'pax.csv'), index=False)

# trip_results_df = pd.concat(trip_results)
# trip_results_df.to_csv(os.path.join(folder_path, 'trips.csv'), index=False)

# state_results_df = pd.concat(state_results)
# state_results_df.to_csv(os.path.join(folder_path, 'state.csv'), index=False)

# idle_results_df = pd.concat(idle_results)
# idle_results_df.to_csv(os.path.join(folder_path, 'idle.csv'), index=False)