import pandas as pd
from objects import Stop, Route, Vehicle, Driver
from params import *

def load_data():
    routes_df = pd.read_csv(PATH_ROUTES)
    stops_df = pd.read_csv(PATH_STOPS)
    
    stops = {}
    for index, row in stops_df.iterrows():
        stops[row['stop_id']] = Stop(row['stop_id'], row['pax_per_hr'])
    
    routes = []
    for route_id in routes_df['route_id'].unique():
        route_df = routes_df[routes_df['route_id']==route_id]
        for direction_id in (0,1):
            route_dir_df = route_df[route_df['direction_id']==direction_id]
            route_id = str(route_dir_df['route_id'].iloc[0]) + '-' + str(direction_id)
            route_stops = route_dir_df['stop_id'].to_list()
            route_stops = route_dir_df['stop_id'].to_list()
            routes.append(Route(route_id, route_stops))
        # route_stops = [stops[stop_id] for stop_id in row['stops'].split(',')]
        # sequence = [stops[stop_id] for stop_id in row['sequence'].split(',')]
        # routes.append(Route(row['route_id'], route_stops, sequence))
    
    return routes, stops

def setup_simulation(env, routes, stops):
    vehicles = []
    drivers = []
    for route in routes:
        vehicle = Vehicle(f'vehicle_{route.id}', route, capacity=40)
        vehicles.append(vehicle)
        driver = Driver(f'driver_{route.id}', vehicle)
        drivers.append(driver)
        env.process(vehicle_process(env, vehicle, stops))
    
    return vehicles, drivers

def vehicle_process(env, vehicle, stops):
    while True:
        for stop in vehicle.route.sequence:
            vehicle.location = stop
            print(f'{env.now}: Vehicle {vehicle.id} arrived at {stop.name}')
            yield env.timeout(10)  # Simulate time spent at stop
            handle_passengers(env, vehicle, stop)
        yield env.timeout(30)  # Simulate time to next stop

def handle_passengers(env, vehicle, stop):
    # Implement pick-up and drop-off logic
    pass
