import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
# Change default DPI for all plots
mpl.rcParams['figure.dpi'] = 150


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
    df[field_name] = df[field_name].astype(str)
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