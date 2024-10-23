import pandas as pd
from params import *
import numpy as np
import matplotlib.pyplot as plt


def get_poisson_arrival_times(arrival_rate, total_time):
    """
    Given an arrival rate and a time interval, returns the times of arrival 
    based on an exponential distribution.
    
    Parameters:
    - arrival_rate (float): The average arrival rate (arrivals per hour).
    - total_time (float): Time in hours.
    
    Returns:
    - list: A list of arrival times (in seconds) during the time interval.
    """
    
    # Convert arrival rate to per second (arrival_rate per hour -> per second)
    arrival_rate_per_sec = arrival_rate / 3600  # 3600 seconds in an hour
    
    # Convert total time from hours to seconds
    total_time_sec = total_time * 3600
    
    # Generate Poisson-distributed number of arrivals
    expected_arrivals = arrival_rate_per_sec * total_time_sec
    num_arrivals = np.random.poisson(expected_arrivals)
    
    # Generate inter-arrival times using the exponential distribution (in seconds)
    inter_arrival_times = np.random.exponential(1 / arrival_rate_per_sec, num_arrivals)
    
    # Cumulatively sum to get the actual arrival times
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Filter out arrival times that exceed the total_time_sec window
    return arrival_times[arrival_times <= total_time_sec].round()

def find_closest_vehicle(vehicles, current_time):
    """
    Finds the index of the vehicle whose event['next']['time'] is closest to the current time.

    Parameters:
    - vehicles (list): A list of vehicle objects, each having an attribute 'event', 
                       which is a dict containing 'next' -> 'time'.
    - current_time (float): The current time to compare the event times with.

    Returns:
    - int: The index of the vehicle with the closest event['next']['time'].
    """
    closest_index = None
    min_time_diff = float('inf')  # Set to infinity initially
    
    for i, vehicle in enumerate(vehicles):
        next_event_time = vehicle.event['next']['time']
        time_diff = abs(next_event_time - current_time)
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_index = i
    
    return closest_index

def pax_activity(vehicle, route, static_dwell, dynamic_dwell, time_now, is_flex):
    """
    Simulates passenger activity at a stop by:
    1. Depositing passengers whose destination is the current stop.
    2. Removing those passengers from the vehicle.
    3. Checking if there are passengers waiting at the current stop, and moving them onto the vehicle.
    4. Counting the number of boardings and alightings.
    5. Applying dwell time based on the number of passengers handled.
    
    Parameters:
    - vehicle (object): The vehicle object, which contains:
        - vehicle.pax: list of passenger objects currently on the vehicle.
        - vehicle.stop: the current stop the vehicle is at.
        - vehicle.direction: direction of travel ('inbound' or 'outbound').
    - route (object): The route object, which contains:
        - route.stops[direction][stop].pax: passengers waiting at the stop.
        - route.archived_pax: list to archive passengers who reach their destination.
    
    Returns:
    - dwell_time (int): The calculated dwell time based on the number of passengers boarding or alighting.
    """
    boardings = 0
    alightings = 0
    wait_time_accumulator = 0
    
    departing_pax = []
    stop_idx = vehicle.event['next']['stop']

    for pax in vehicle.pax:
        if pax.destination == stop_idx:
            pax.alight_time = time_now
            route.archived_pax.append(pax)
            departing_pax.append(pax)
            alightings += 1

    for pax in departing_pax:
        vehicle.pax.remove(pax)

    stop_pax = route.stops[vehicle.direction][stop_idx].active_pax
    for pax in stop_pax:
        pax.boarding_time = time_now
        vehicle.pax.append(pax)
        wait_time_accumulator += (time_now - pax.arrival_time)
        boardings += 1

    route.stops[vehicle.direction][stop_idx].active_pax = []

    total_pax = boardings + alightings
    dwell_time = ((total_pax > 0) * static_dwell) + total_pax * dynamic_dwell

    ## add records
    vehicle.event_hist['boardings'].append(boardings)
    vehicle.event_hist['alightings'].append(alightings)
    vehicle.event_hist['load'].append(len(vehicle.pax))
    vehicle.event_hist['departure_time'].append(vehicle.event_hist['arrival_time'][-1] + dwell_time)

    ## update route wait time
    if not is_flex:
        route.inter_event['fixed_wait_time'] += wait_time_accumulator ## hours
        route.inter_event['fixed_boardings'] += boardings
    return dwell_time

def get_observation(vehicle: object, route: object, control_stops: list):
    ## get stop, direction, and next flex stop of vehicle
    stop_idx, direction = vehicle.get_location()
    if stop_idx not in control_stops:
        return None
    flex_stop_idx = stop_idx + 1
    stop = route.stops[direction][stop_idx]
    flex_stop = route.stops[direction][flex_stop_idx]
    n_flex_pax = route.get_n_waiting_pax(flex_stop_idx, direction)

    ## check conditions
    is_departing = vehicle.event['next']['type'] == 'depart'
    flex_pax_waiting = len(flex_stop.active_pax) > 0
    not_first_arrival = len(stop.last_arrival_time) > 1

    if is_departing and flex_pax_waiting and not_first_arrival:
        observation = [
            np.int32(stop_idx), 
            np.int32(n_flex_pax), 
            np.int32(len(vehicle.pax)), 
            np.float32(stop.get_latest_headway()), 
            np.float32(vehicle.get_latest_delay())
        ]
        return observation
    return None

def lognormal_sample(stats):
    mean = stats['mean']
    std = stats['std']
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    return round(np.random.lognormal(mu, sigma),0)

def get_pax_hist(route, flex_stops, include_denied=False):
    pax_hist = {'direction': [], 'origin': [], 'destination': [], 'arrival_time': [], 'boarding_time': [], 'alight_time': []}

    for pax in route.archived_pax:
        pax_hist['direction'].append(pax.direction)
        pax_hist['origin'].append(pax.origin)
        pax_hist['destination'].append(pax.destination)
        pax_hist['arrival_time'].append(pax.arrival_time)
        pax_hist['boarding_time'].append(pax.boarding_time)
        pax_hist['alight_time'].append(pax.alight_time)

    if include_denied:
        for pax in route.denied_flex_pax:
            pax_hist['direction'].append(pax.direction)
            pax_hist['origin'].append(pax.origin)
            pax_hist['destination'].append(pax.destination)
            pax_hist['arrival_time'].append(pax.arrival_time)
            pax_hist['boarding_time'].append(np.nan)
            pax_hist['alight_time'].append(np.nan)

    pax_df = pd.DataFrame(pax_hist)
    pax_df['wait_time'] = pax_df['boarding_time']-pax_df['arrival_time']

    pax_df['flex'] = 0
    pax_df.loc[pax_df['origin'].isin(flex_stops), 'flex'] = 1
    return pax_df

def get_vehicle_history(vehicle_list, flex_stop_list):
    """
    Generates a DataFrame containing the event history of each vehicle.

    Parameters:
    - vehicle_list (list): A list of vehicle objects, each having an attribute 'event_hist'.
    - flex_stop_list (list): A list of stops considered as flexible stops.

    Returns:
    - pd.DataFrame: A DataFrame containing the event history of all vehicles, 
                    with additional columns for vehicle ID, headway, and flex stop indicator.
    """
    veh_hist = []
    for veh in vehicle_list:
        df = pd.DataFrame(veh.event_hist)
        df['veh_id'] = veh.idx
        veh_hist.append(df)
    veh_df = pd.concat(veh_hist, ignore_index=True)
    veh_df['headway'] = veh_df.groupby(['direction', 'stop'])['arrival_time'].transform(lambda x: x.sort_values().diff())

    veh_df['flex'] = 0
    veh_df.loc[veh_df['stop'].isin(flex_stop_list), 'flex'] = 1
    return veh_df

def convert_duration_string_to_minutes(duration_str):
    ## receive HHhMM string and convert to minutes
    hours, minutes = duration_str.split('h')
    return int(hours) * 60 + int(minutes)

def pct_change(val_from, val_to, decimals=2):
    return round((val_to - val_from) / val_from, decimals)

def get_min_pax_threshold(delay):
    if delay < MIN_PAX_THRESHOLDS[0][0]:
        return MIN_PAX_THRESHOLDS[0][1]
    elif delay < MIN_PAX_THRESHOLDS[1][0]:
        return MIN_PAX_THRESHOLDS[1][1]
    elif delay < MIN_PAX_THRESHOLDS[2][0]:
        return MIN_PAX_THRESHOLDS[2][1]
    else:
        return MIN_PAX_THRESHOLDS[3][1]
    
def plot_min_pax_threshold():
    fig, axs = plt.subplots(figsize=(4, 3))
    delays = np.arange(0, 10, 0.1)
    thresholds = [get_min_pax_threshold(delay) for delay in delays]
    axs.plot(delays, thresholds)
    axs.set_xlabel('Delay (minutes)')
    axs.set_ylabel('Minimum Pax Threshold')
    axs.set_title('Minimum Pax Threshold vs. Delay')
    plt.show()

# Example usage:
# To get the bin index for 'headway' with an actual value of 7:
# bin_index = get_bin_index('headway', 7)   # Returns 1, because 7 falls into the 5-10 range
