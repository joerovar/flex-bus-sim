import pandas as pd
from params import *
import numpy as np


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

def pax_activity(vehicle, route, static_dwell, dynamic_dwell, time_now):
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
        boardings += 1

    route.stops[vehicle.direction][stop_idx].active_pax = []

    total_pax = boardings + alightings
    dwell_time = ((total_pax > 0) * static_dwell) + total_pax * dynamic_dwell

    ## add records
    vehicle.event_hist['boardings'].append(boardings)
    vehicle.event_hist['alightings'].append(alightings)
    vehicle.event_hist['load'].append(len(vehicle.pax))
    vehicle.event_hist['departure_time'].append(vehicle.event_hist['arrival_time'][-1] + dwell_time)
    
    return dwell_time

def check_control_conditions(vehicle, control_stops):
    is_in_control_stop = vehicle.event['next']['stop'] in control_stops
    is_departing = vehicle.event['next']['type'] == 'depart'
    return (is_in_control_stop and is_departing)

def lognormal_sample(stats):
    mean = stats['mean']
    std = stats['std']
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    return round(np.random.lognormal(mu, sigma),0)


def get_pax_hist(route):
    pax_hist = {'direction': [], 'origin': [], 'destination': [], 'arrival_time': [], 'boarding_time': [], 'alight_time': []}

    for pax in route.archived_pax:
        pax_hist['direction'].append(pax.direction)
        pax_hist['origin'].append(pax.origin)
        pax_hist['destination'].append(pax.destination)
        pax_hist['arrival_time'].append(pax.arrival_time)
        pax_hist['boarding_time'].append(pax.boarding_time)
        pax_hist['alight_time'].append(pax.alight_time)

    pax_df = pd.DataFrame(pax_hist)
    pax_df['wait_time'] = pax_df['boarding_time']-pax_df['arrival_time']
    return pax_df

def get_veh_hist(route):
    veh_hist = []
    for veh in route.vehicles:
        df = pd.DataFrame(veh.event_hist)
        df['veh_id'] = veh.idx
        veh_hist.append(df)
    veh_df = pd.concat(veh_hist, ignore_index=True)
    veh_df['headway'] = veh_df.groupby(['direction', 'stop'])['arrival_time'].transform(lambda x: x.sort_values().diff())
    return veh_df
