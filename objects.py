from helpers import *

class Passenger:
    def __init__(self, origin, dest, arrival_time) -> None:
        self.origin = origin
        self.dest = dest
        self.arrival_time = arrival_time
        self.boarding_time = None
        self.alight_time = None


class Stop:
    def __init__(self, idx) -> None:
        self.idx = idx
        self.active_pax = []
        self.inactive_pax_arrival_times = np.array([]) ## to easily filter
    
    def move_to_active_pax(self, time_now):
        tmp_times = self.inactive_pax_arrival_times
        any_pax_times = list(tmp_times[tmp_times <= time_now])
        if len(any_pax_times):
            for pax_time in any_pax_times:
                dest = np.random.randint([i for i in range(self.idx+1, N_STOPS) if i not in FLEX_STOPS])
                self.active_pax.append(Passenger(self.idx, dest, pax_time))
            self.inactive_pax_arrival_times = tmp_times[tmp_times>time_now]


class Schedule:
    def __init__(self) -> None:
        schedule_headway = 10 ## minutes
        n_trips = 100
        half_cycle_time = 10
        self.deps = {
            'out': [i*schedule_headway*60 for i in range(n_trips)],
            'in': [half_cycle_time*60 + i*schedule_headway*60 for i in range(n_trips)]
        }

class RouteManager:
    def __init__(self) -> None:
        # self.n_stops = 7
        # self.flex_stops = [2, 4]
        n_vehicles = 2

        self.stops = {
            'in': [Stop(i) for i in range(N_STOPS)],
            'out': [Stop(i) for i in range(N_STOPS)]
        }

        self.arr_rates = {
            'flex': 6,
            'fixed': 10,
            'terminal': 20
        }
        
        self.vehicles = [Vehicle(i, i) for i in range(n_vehicles)]

        ## used to count the trip for the vehicle
        self.max_trip_nr = {
            'in': n_vehicles,
            'out': n_vehicles
        }

        self.archived_pax = []
        self.schedule = Schedule()

    def load_all_pax(self):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)):
                if i == 0: ## terminal
                    arr_rate = self.arr_rates['terminal']
                if i in FLEX_STOPS:
                    arr_rate = self.arr_rates['flex']
                if i > 0 and i not in FLEX_STOPS:
                    arr_rate = self.arr_rates['fixed']
                self.stops[direction].inactive_pax_arrival_times = get_poisson_arrival_times(arr_rate, MAX_TIME)
    
    def get_active_pax(self, time_now):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)):
                self.stops[direction][i].move_to_active_pax(time_now)



class Vehicle:
    def __init__(self, trip_nr_in, trip_nr_out):
        self.pax = []
        self.direction = 'out' 
        self.trip_nr = {
            'in': trip_nr_in, ## starting from outbound
            'out': trip_nr_out
        }
        self.event = {
            'last': {'time': None, 'type': None, 'stop': None},
            'next': {'time': None, 'type': None, 'stop': None}
        }
    
    def start(self, schedule):
        self.direction = 'out'
        ## next event is outbound departure
        trip_nr = self.trip_nr[self.direction]
        self.event['next']['time'] = schedule.deps[self.direction][trip_nr]
        self.event['next']['stop'] = 0
        self.event['next']['type'] = 'arrive'
    
    def layover(self, route, dwell_time, time_now):
        finish_time = time_now + dwell_time
        
        next_direction = 'in' if self.direction == 'out' else 'out'
        next_trip_nr = route.max_trip_nr[next_direction] + 1
        self.trip_nr[next_direction] = next_trip_nr

        ## add +1 to the latest trip number
        route.max_trip_nr[next_direction] += 1

        ## get next scheduled time
        next_schedule_time = route.schedule.deps[next_direction][next_trip_nr]

        ## the next event time is the latest between schedule and finish time
        self.event['next']['time'] = max(next_schedule_time, finish_time)
        self.event['next']['type'] = 'arrive'
        self.event['next']['stop'] = 0
        self.direction = next_direction
        
    
    def arrive_station(self, route):
        time_now = self.event['next']['time']
        dwell_time = pax_activity(self, route)

        self.event['last']['time'] = self.event['next']['time']
        self.event['last']['type'] = 'arrive'

        if self.event['next']['stop'] < N_STOPS - 2:
            self.event['next']['time'] = time_now + dwell_time
            self.event['next']['type'] = 'depart'
        else:
            self.layover(route, dwell_time, time_now)

    
    def depart_station(self, skip_flex=False):
        time_now = self.event['next']['time']
        ## set destination
        self.event['last']['time'] = time_now
        self.event['last']['type'] = 'depart'
        self.event['last']['stop'] = self.event['next']['stop']
        self.event['next']['type'] = 'arrive'

        if skip_flex:
            ## stop at 
            self.event['next']['stop'] += 2
        else:
            self.event['next']['stop'] += 1

        orig_flex = self.event['last']['stop'] in FLEX_STOPS
        dest_flex = self.event['next']['stop'] in FLEX_STOPS
        if (orig_flex or dest_flex):
            run_time = 1.5 * 60
        else:
            run_time = 2 * 60
        
        self.event['next']['time'] = time_now + run_time

class EventManager:
    def __init__(self) -> None:
        start_time = 0
        self.timestamps = [start_time]
    
    def start_vehicles(self, route):
        for vehicle in route.vehicles:
            vehicle.start(route.schedule)

    def process_event(self):
        return
    
    def step(self, route, action=None):
        veh_idx = find_closest_vehicle(route.vehicles, self.timestamps[-1])
        route.get_active_pax(self.timestamps[-1])

        ## add new time to list of timestamps
        self.timestamps.append(route.vehicles[veh_idx].event['next']['time'])

        if route.vehicles[veh_idx].event['next']['type'] == 'arrive':
            route.vehicles[veh_idx].arrive_station(route)
        
        if route.vehicles[veh_idx].event['next']['type'] == 'depart':
            route.vehicles[veh_idx].depart_station(route)

