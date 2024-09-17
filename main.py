import simpy
from helpers import load_data, setup_simulation

if __name__ == '__main__':
    env = simpy.Environment()
    routes, stops = load_data()
    # vehicles, drivers = setup_simulation(env, routes, stops)
    # env.run(until=20)
