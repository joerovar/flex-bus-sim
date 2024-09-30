from helpers import *

class Passenger:
    def __init__(self, origin, destination, arrival_time, direction) -> None:
        self.origin = origin
        self.destination = destination
        self.direction = direction
        self.arrival_time = arrival_time
        self.boarding_time = None
        self.alight_time = None


class Stop:
    def __init__(self, idx, direction) -> None:
        self.idx = idx
        self.active_pax = []
        self.inactive_pax_arrival_times = np.array([]) ## to easily filter
        self.direction = direction
    
    def move_to_active_pax(self, time_now):
        tmp_times = self.inactive_pax_arrival_times
        any_pax_times = list(tmp_times[tmp_times <= time_now])
        if len(any_pax_times):
            for pax_time in any_pax_times:
                dest = np.random.choice([i for i in range(self.idx+1, N_STOPS) if i not in FLEX_STOPS])
                self.active_pax.append(Passenger(self.idx, dest, pax_time, self.direction))
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
        self.stops = {
            'in': [Stop(i, 'in') for i in range(N_STOPS)],
            'out': [Stop(i, 'out') for i in range(N_STOPS)]
        }
        
        self.vehicles = [Vehicle(i) for i in range(N_VEHICLES)]

        ## used to track the latest assigned trip number
        self.trip_counter = {
            'in': 0,
            'out': 0
        }

        self.archived_pax = []
        self.schedule = Schedule()

    def load_all_pax(self):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)-1):
                if i == 0: ## terminal
                    arr_rate = ARRIVAL_RATES['terminal']
                if i in FLEX_STOPS:
                    arr_rate = ARRIVAL_RATES['flex']
                if i > 0 and i not in FLEX_STOPS:
                    arr_rate = ARRIVAL_RATES['fixed']
                self.stops[direction][i].inactive_pax_arrival_times = get_poisson_arrival_times(arr_rate, MAX_TIME_HOURS)
    
    def get_active_pax(self, time_now):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)):
                self.stops[direction][i].move_to_active_pax(time_now)
    
    def assign_next_trip(self, direction):
        ## update trip counter for next direction
        self.trip_counter[direction] += 1

        ## get next scheduled departure for the specified direction
        trip_idx = self.trip_counter[direction] - 1 ## minus 1 because departures index start from zero
        schd_time = self.schedule.deps[direction][trip_idx]
        return schd_time


class Vehicle:
    def __init__(self, idx):
        self.idx = idx
        self.pax = []
        self.direction = None
        self.trip_counter = {
            'in': 0, ## starting from outbound
            'out': 0
        }
        self.event = {
            'last': {'time': None, 'type': None, 'stop': None},
            'next': {'time': None, 'type': None, 'stop': None}
        }

        ## records
        self.event_hist = {'direction': [], 'stop': [], 'arrival_time': [], 'departure_time': [], 'load': [],
                           'boardings': [], 'alightings': []}
    
    def start(self, route):
        self.direction = 'out'

        ## update route trip coutner and get the scheduled departure
        next_schd_time = route.assign_next_trip(self.direction)
        ## trip counter
        self.trip_counter[self.direction] += 1

        self.event['next']['time'] = next_schd_time
        self.event['next']['stop'] = 0
        self.event['next']['type'] = 'arrive'
    
    def layover(self, route, dwell_time, time_now):
        finish_time = time_now + dwell_time
        
        next_direction = 'in' if self.direction == 'out' else 'out'
        
        ## update route trip coutner and get the scheduled departure
        next_schd_time = route.assign_next_trip(next_direction)

        ## update vehicle trip counter
        self.trip_counter[next_direction] += 1

        ## the next event time is the latest between schedule and finish time
        self.event['next']['time'] = max(next_schd_time, finish_time)
        self.event['next']['type'] = 'arrive'
        self.event['next']['stop'] = 0
        self.direction = next_direction
        
    
    def arrive_station(self, route):
        time_now = self.event['next']['time']
        
        ## append records
        self.event_hist['direction'].append(self.direction)
        self.event_hist['stop'].append(self.event['next']['stop'])
        self.event_hist['arrival_time'].append(time_now)
        
        ## process
        dwell_time = pax_activity(self, route, STATIC_DWELL, DYNAMIC_DWELL, time_now)

        self.event['last']['time'] = self.event['next']['time']
        self.event['last']['type'] = 'arrive'
        self.event['last']['stop'] = self.event['next']['stop']

        if self.event['next']['stop'] == N_STOPS - 1:
            ## continue onto layover
            self.layover(route, dwell_time, time_now)
        else:
            self.event['next']['time'] = time_now + dwell_time
            self.event['next']['type'] = 'depart'

    
    def depart_station(self, skip_flex=False):
        time_now = self.event['next']['time']

        ## set destination
        self.event['last']['time'] = time_now
        self.event['last']['type'] = 'depart'
        self.event['last']['stop'] = self.event['next']['stop']
        self.event['next']['type'] = 'arrive'

        stop = self.event['next']['stop']

        if skip_flex and (stop in CONTROL_STOPS):
            ## stop at 
            self.event['next']['stop'] += 2
        else:
            self.event['next']['stop'] += 1

        orig_flex = self.event['last']['stop'] in FLEX_STOPS
        dest_flex = self.event['next']['stop'] in FLEX_STOPS

        segment_type = 'flex' if (orig_flex or dest_flex) else 'fixed'
        run_time = lognormal_sample(SEGMENT_TIMES[segment_type])
        
        self.event['next']['time'] = time_now + run_time

class EventManager:
    def __init__(self) -> None:
        start_time = 0
        self.timestamps = [start_time]

        self.done = 0
        self.requires_control = 0
        self.veh_idx = None
    
    def start_vehicles(self, route):
        for vehicle in route.vehicles:
            vehicle.start(route)

    def step(self, route, action=None):
        if (action is not None) and (self.veh_idx is not None):
            route.vehicles[self.veh_idx].depart_station(route)
            return self.step(route)
        
        self.veh_idx = find_closest_vehicle(route.vehicles, self.timestamps[-1])

        ## add new time to list of timestamps
        self.timestamps.append(route.vehicles[self.veh_idx].event['next']['time'])

        time_now = self.timestamps[-1]
        
        if time_now > MAX_TIME_HOURS*60*60:
            return (1, None, [])
        
        ## get all passengers up to the current time
        route.get_active_pax(time_now)

        requires_control = check_control_conditions(route.vehicles[self.veh_idx], CONTROL_STOPS)

        if requires_control:
            flex_stop_idx = route.vehicles[self.veh_idx].event['next']['stop'] + 1
            direction = route.vehicles[self.veh_idx].direction
            flex_stop = route.stops[direction][flex_stop_idx]
            n_pax = len(flex_stop.active_pax)
            return (0, None, [n_pax])
        else:
            if route.vehicles[self.veh_idx].event['next']['type'] == 'arrive':
                route.vehicles[self.veh_idx].arrive_station(route)
                return self.step(route)
            
            if route.vehicles[self.veh_idx].event['next']['type'] == 'depart':
                route.vehicles[self.veh_idx].depart_station(route)
                return self.step(route)

