import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Change default DPI for all plots
mpl.rcParams['figure.dpi'] = 150

def tabulate_improvements(state, idle, pax, trips, base_scenario='DN', flex_stops=[]):
    # Group and aggregate results for each DataFrame
    n_lates = state.groupby(['scenario'])['reward_3'].sum()
    denied_riders = state.groupby(['scenario'])['reward_1'].sum()
    idle_sum = (idle.groupby(['scenario'])['idle_time'].sum()/60/60).round(2)
    wait_time_mean = pax.groupby(['scenario'])['wait_time'].mean().round(0)
    headway_cv = (trips.groupby(['scenario'])['headway'].std() / trips.groupby(['scenario'])['headway'].mean()).round(3)
    avg_load = trips.groupby(['scenario'])['load'].mean().round(2)
    n_fixed_pax = pax[~pax['origin'].isin(flex_stops)].groupby(['scenario']).size()
    n_flex_pax = pax[pax['origin'].isin(flex_stops)].groupby(['scenario']).size()
    n_tot_pax = pax.groupby(['scenario']).size()
    n_trips = trips[trips['stop']==0].groupby(['scenario']).size()

    # Create a DataFrame with all the metrics
    result_df = pd.DataFrame({
        'n_lates': n_lates,
        'idle_time': idle_sum,
        'wait_time': wait_time_mean,
        'headway_cv': headway_cv,
        'load': avg_load,
        'denied_riders': denied_riders,
        'fixed_ridership': n_fixed_pax,
        'flex_ridership': n_flex_pax,
        'tot_ridership': n_tot_pax,
        'n_trips': n_trips
    })
    
    # Calculate percentage change from the base scenario
    pct_change_df = result_df.copy()
    for col in result_df.columns:
        base_value = result_df.loc[base_scenario, col]
        pct_change_df[col] = ((result_df[col] - base_value) / base_value * 100).round(3)
    
    # Return the result DataFrame
    return result_df, pct_change_df


def plot_cumulative_idle_time(df):
    # Sort by time to ensure correct cumulative plotting within each scenario
    df = df.sort_values(by=['scenario', 'time'])
    
    # Create the cumulative sum of idle_time for each episode in each scenario
    df['cumulative_idle_time'] = df.groupby(['scenario'])['idle_time'].cumsum()
    
    # Set up the seaborn grid style
    sns.set(style="whitegrid")
    
    # Create the plot with multiple scenarios
    fig, axs = plt.subplots(figsize=(7, 5))
    
    # Plot time vs cumulative idle time with distribution for each scenario (hue=scenario)
    sns.lineplot(x='time', y='cumulative_idle_time', data=df, hue='scenario', ax=axs)
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel('Avg Cumulative Idle Time')
    plt.title('Avg Cumulative Idle Time over Time by Scenario')
    
    # Display the plot
    plt.show()