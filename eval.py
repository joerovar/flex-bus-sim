from objects import EnvironmentManager
import pandas as pd
import os
from datetime import datetime
from rl_env import *
from itertools import product

## Experimental Design Parameters
MINIMUM_REQUEST_SLOPES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] 

# Float to '00' string
def float_to_string(num):
    # Format as 2 digits with one decimal place, then remove the decimal point
    return f"{num:.1f}".replace(".", "")


def get_heuristic_action(observation, alpha, beta):
    schedule_deviation = observation[3] / 60 # convert to minutes
    n_requests = observation[1]
    min_pax = max(schedule_deviation*beta + alpha, 0)
    if n_requests > min_pax:
        return 1
    else:
        return 0

def evaluate_heuristic(alpha=HEURISTIC_ALPHA, beta=HEURISTIC_BETA,
                       demand_scenario='peak', 
                       n_episodes=30, output_history=False,
                       scenario_name=None):
    history = {'pax': [], 'vehicles': [], 'idle': []}
    rewards_per_episode = []
    results_per_episode = {
        'deviation_opportunities': [],
        'deviations': [],
        'avg_picked_requests': [],
        'early_trips': [],
        'late_trips': []
    }

    ## evaluation
    for i in range(n_episodes):
        env = EnvironmentManager()
        env.start_vehicles()
        env.route.load_all_pax(demand_scenario=demand_scenario)

        observation, reward, done, terminated, info = env.step(action=None)
        while not terminated:
            if not done:
                action = get_heuristic_action(
                    observation, alpha=alpha, beta=beta) # ideal slope
            else:
                action = None
            observation, reward, done, terminated, info = env.step(action=action)

        # update results
        deviation_opps, deviations, avg_picked_requests, early_trips, late_trips = env.get_tracker_info()
        results_per_episode['deviation_opportunities'].append(deviation_opps)
        results_per_episode['deviations'].append(deviations)
        results_per_episode['avg_picked_requests'].append(avg_picked_requests)
        results_per_episode['early_trips'].append(early_trips)
        results_per_episode['late_trips'].append(late_trips)

        if output_history:
            episode_history = env.get_history()
            for key in episode_history:
                episode_history[key]['episode'] = i
                if scenario_name:
                    episode_history[key]['scenario'] = scenario_name
                history[key].append(episode_history[key])

    if output_history:
        for df_key in history:
            history[df_key] = pd.concat(history[df_key])
        return history
    else:
        summary_results = {
            'mean_reward': round(np.nanmean(rewards_per_episode),3),
            'std_reward': round(np.nanstd(rewards_per_episode),3),
            'deviation_opportunities': round(np.mean(results_per_episode['deviation_opportunities']),1),
            'deviations': round(np.mean(results_per_episode['deviations']),1),
            'avg_picked_requests': round(np.mean(results_per_episode['avg_picked_requests']),1),
            'early_trips': round(np.mean(results_per_episode['early_trips']),1),
            'late_trips': round(np.mean(results_per_episode['late_trips']),1)
        }
        return summary_results