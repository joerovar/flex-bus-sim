import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Change default DPI for all plots
mpl.rcParams['figure.dpi'] = 150

def tabulate_improvements(state: pd.DataFrame, idle: pd.DataFrame, 
                          pax: pd.DataFrame, trips: pd.DataFrame, on_time_bounds: list,
                          base_scenario: str = 'DN',flex_stops: list = []) -> tuple:
    # Group and aggregate results for each DataFrame
    n_denied = pax[pax['boarding_time'].isna()].groupby(['scenario']).size()
    idle_sum = (idle.groupby(['scenario'])['idle_time'].sum()/60/60).round(2)
    wait_time_mean = pax.groupby(['scenario'])['wait_time'].mean().round(0)
    fixed_trip_stops = trips[~trips['stop'].isin(flex_stops)]
    headway_cv = (fixed_trip_stops.groupby(['scenario'])['headway'].std() / fixed_trip_stops.groupby(['scenario'])['headway'].mean()).round(3)
    avg_load = trips.groupby(['scenario'])['load'].mean().round(2)
    n_fixed_pax = pax[(~pax['origin'].isin(flex_stops)) & (pax['boarding_time'].notna())].groupby(['scenario']).size()
    n_flex_pax = pax[(pax['origin'].isin(flex_stops)) & (pax['boarding_time'].notna())].groupby(['scenario']).size()
    n_tot_pax = pax[pax['boarding_time'].notna()].groupby(['scenario']).size()
    n_trips = trips[trips['stop']==0].groupby(['scenario']).size()
    on_time_trips = trips[(trips['delay'].between(*on_time_bounds)) & (trips['stop']==0)].groupby(['scenario']).size()
    trips['delay'] = trips['arrival_time'] - trips['scheduled_time']
    average_delay_by_group = trips[~trips['stop'].isin(flex_stops)].groupby('scenario')['delay'].mean().round(0)
    n_deviations = state.groupby(['scenario'])['action'].sum()
    avg_reward = state.groupby(['scenario'])['reward'].mean().round(3)
    avg_episode_reward = state.groupby(['scenario', 'episode'])['reward'].sum()
    avg_episode_reward = avg_episode_reward.groupby('scenario').mean().round(3)
    ## get mean delay where delay is the difference between arrival time and scheduled time 
    

    # Create a DataFrame with all the metrics
    result_df = pd.DataFrame({
        # 'n_lates': n_lates,
        'idle_time': idle_sum,
        'wait_time': wait_time_mean,
        'headway_cv': headway_cv,
        'load': avg_load,
        'n_denied_riders': n_denied,
        'fixed_ridership': n_fixed_pax,
        'flex_ridership': n_flex_pax,
        'tot_ridership': n_tot_pax,
        'n_trips': n_trips,
        'on_time_trips': on_time_trips,
        'avg_delay': average_delay_by_group,
        'n_deviations': n_deviations,
        'avg_reward': avg_reward,
        'avg_episode_reward': avg_episode_reward
    })
    result_df['on_time_rate'] = (result_df['on_time_trips'] / result_df['n_trips'] * 100).round(2)
    result_df['served_rate'] = 100 - result_df['n_denied_riders'] / (result_df['n_denied_riders']+result_df['flex_ridership']) * 100
    result_df['served_rate'] = result_df['served_rate'].fillna(0.0).round(2)
    # Calculate percentage change from the base scenario
    pct_change_df = result_df.copy()
    for col in result_df.columns:
        base_value = result_df.loc[base_scenario, col]
        pct_change_df[col] = ((result_df[col] - base_value) / base_value * 100).round(3)
    
    # Return the result DataFrame
    return result_df, pct_change_df


def plot_cumulative_idle_time(df, figsize=(4,3)):
    # Sort by time to ensure correct cumulative plotting within each scenario
    df = df.sort_values(by=['scenario', 'time'])
    
    # Create the cumulative sum of idle_time for each episode in each scenario
    df['cumulative_idle_time'] = df.groupby(['scenario'])['idle_time'].cumsum()
    
    # Set up the seaborn grid style
    sns.set(style="whitegrid")
    
    # Create the plot with multiple scenarios
    fig, axs = plt.subplots(figsize=figsize)
    
    # Plot time vs cumulative idle time with distribution for each scenario (hue=scenario)
    sns.lineplot(x='time', y='cumulative_idle_time', data=df, hue='scenario', ax=axs)
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Avg Cumulative Idle Time')
    plt.title('Avg Cumulative Idle Time over Time by Scenario')
    
    # Display the plot
    plt.show()


def create_field_from_list_column(df, list_index, new_field_name, field_name='observation'):
    df[new_field_name] = df[field_name].apply(lambda x: float(x.split(',')[list_index].strip('[]')))

import numpy as np

def plot_exponential_decay_by_factor():
    # Create a range of factors from 0.1 to 1.0
    factors = np.linspace(0.005, 0.007, 5)
    
    # Create a range of time values from 0 to 450
    time_values = np.arange(0, 500)
    
    # Set up the seaborn darkgrid style
    sns.set_style("darkgrid")
    
    # Create a plot with the time values on the x-axis and the exponential decay for each factor
    fig, axs = plt.subplots(figsize=(4, 2.5))
    for factor in factors:
        decay_values = np.exp(-factor * time_values)
        sns.lineplot(x=time_values, y=decay_values, ax=axs, label=f'Factor: {factor:.3f}')
    axs.set_xlabel('Time')
    axs.set_ylabel('Exponential Decay')
    axs.set_title('Exponential Decay by Factor')
    axs.legend(title='Decay Factor')
    axs.set_ylim(0,1)
    plt.show()

## OD_MATRIX IS A LIST OF LISTS OF INTEGERS
## GIVE ME A HEATMAP OUT OF THEM
def get_heatmap(od_matrix, title):
    plt.figure(figsize=(4, 4))
    sns.heatmap(od_matrix, annot=True, cmap="YlGnBu")
    plt.title(title)