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
        self.last_arrival_time = [-SCHEDULE_HEADWAY*60]
    
    def move_to_active_pax(self, time_now):
        for dest_idx in range(len(self.inactive_pax_arrival_times)):
            tmp_times = np.array(self.inactive_pax_arrival_times[dest_idx])
            any_pax_times = list(tmp_times[tmp_times <= time_now])
            if len(any_pax_times):
                for pax_time in any_pax_times:
                    self.active_pax.append(Passenger(self.idx, dest_idx, pax_time, self.direction))
                self.inactive_pax_arrival_times[dest_idx] = list(tmp_times[tmp_times>time_now])

    def remove_long_wait_pax(self, time_now):
        ## only flex pax
        long_wait_pax = []
        for pax in self.active_pax:
            if time_now - pax.arrival_time > MAX_WAIT_TIME_FLEX * 60:
                long_wait_pax.append(pax)
        for pax in long_wait_pax:
            self.active_pax.remove(pax)
        return long_wait_pax
    
    def get_latest_headway(self):
        return self.last_arrival_time[-1] - self.last_arrival_time[-2]


class Schedule:
    def __init__(self) -> None:
        self.deps = {
            'out': [i*SCHEDULE_HEADWAY*60 for i in range(N_TRIPS)],
            'in': [HALF_CYCLE_TIME*60 + i*SCHEDULE_HEADWAY*60 for i in range(N_TRIPS)]
        }
        self.scheduled_stop_times = SCHEDULED_STOP_TIMES

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

        self.idle_time = {
            'time': [],
            'idle_time': []
        }

        self.archived_pax = []
        self.schedule = Schedule()
        self.denied_flex_pax = []
        self.inter_event = {
            'denied': 0,
            'fixed_wait_time': 0,
            'late': 0,
            'fixed_boardings': 0
        }

    def load_all_pax(self):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)-1):
                inter_arrival_times = []
                for j in range(len(set_stops)):
                    od_rate = OD_MATRIX[i][j]
                    if od_rate > 0:
                        arrival_times = list(get_poisson_arrival_times(od_rate, MAX_TIME_HOURS))
                        inter_arrival_times.append(arrival_times)
                    else:
                        inter_arrival_times.append([])
                self.stops[direction][i].inactive_pax_arrival_times = inter_arrival_times
    
    def get_active_pax(self, time_now):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)):
                ## discard first if flex
                if i in FLEX_STOPS and REMOVE_LONG_WAIT_FLEX:
                    denied_flex_pax = self.stops[direction][i].remove_long_wait_pax(time_now)
                    self.denied_flex_pax += denied_flex_pax
                    self.inter_event['denied'] += len(denied_flex_pax)

                self.stops[direction][i].move_to_active_pax(time_now)

    def get_scheduled_time(self, stop_idx, trip_idx, direction):
        schd_depart = self.schedule.deps[direction][trip_idx]
        return schd_depart + self.schedule.scheduled_stop_times[stop_idx]
    
    def assign_next_trip(self, direction):
        ## update trip counter for next direction
        self.trip_counter[direction] += 1

        ## get next scheduled departure for the specified direction
        trip_idx = self.trip_counter[direction] - 1 ## minus 1 because departures index start from zero
        schd_time = self.schedule.deps[direction][trip_idx]
        return schd_time, trip_idx

    def get_reward(self, event):
        reward_1 = self.inter_event['denied'] * REWARD_WEIGHTS['denied']
        
        avg_wait_time = self.inter_event['fixed_wait_time'] / np.max([self.inter_event['fixed_boardings'], 1]) 
        excess_wait_time = (avg_wait_time/60) - (SCHEDULE_HEADWAY/2)
        reward_2 = max(0, round(excess_wait_time,0)*REWARD_WEIGHTS['fixed_wait_time'])
        
        reward_3 = self.inter_event['late'] * REWARD_WEIGHTS['late']
        
        tot_reward = reward_1 + reward_2 + reward_3

        event.state_hist['reward_1'].append(reward_1)
        event.state_hist['reward_2'].append(reward_2)
        event.state_hist['reward_3'].append(reward_3)
        event.state_hist['tot_reward'].append(tot_reward)
        return tot_reward


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
        self.event_hist = {'direction': [], 'trip_id': [], 'stop': [], 'arrival_time': [], 
                           'departure_time': [], 'load': [],
                           'boardings': [], 'alightings': [], 'scheduled_time': []}
        self.trip_idx = None
    
    def start(self, route):
        self.direction = 'out'

        ## update route trip coutner and get the scheduled departure
        next_schd_time, trip_idx = route.assign_next_trip(self.direction)
        self.trip_idx = trip_idx

        ## trip counter
        self.trip_counter[self.direction] += 1

        self.event['next']['time'] = next_schd_time
        self.event['next']['stop'] = 0
        self.event['next']['type'] = 'arrive'
    
    def layover(self, route, dwell_time, time_now):
        finish_time = time_now + dwell_time
        
        next_direction = 'in' if self.direction == 'out' else 'out'
        
        ## update route trip coutner and get the scheduled departure
        next_schd_time, trip_idx = route.assign_next_trip(next_direction)
        self.trip_idx = trip_idx

        ## update vehicle trip counter
        self.trip_counter[next_direction] += 1

        ## the next event time is the latest between schedule and finish time
        self.event['next']['time'] = max(next_schd_time, finish_time)
        self.event['next']['type'] = 'arrive'
        self.event['next']['stop'] = 0
        self.direction = next_direction

        ## update the late arrivals counter
        if self.event['next']['time'] > next_schd_time + SCHEDULE_TOLERANCE:
            route.inter_event['late'] += 1
        
        ## update idle time tracker
        idle_time = max(0, next_schd_time - finish_time)
        route.idle_time['time'].append(time_now)
        route.idle_time['idle_time'].append(idle_time)
        
    
    def arrive_at_stop(self, route):
        time_now = self.event['next']['time']
        
        ## append records
        self.event_hist['trip_id'].append(self.trip_idx)
        self.event_hist['direction'].append(self.direction)
        self.event_hist['stop'].append(self.event['next']['stop'])
        self.event_hist['arrival_time'].append(time_now)
        self.event_hist['scheduled_time'].append(route.get_scheduled_time(self.event['next']['stop'], self.trip_idx, self.direction))

        
        ## process
        is_flex = self.event['next']['stop'] in FLEX_STOPS
        dwell_time = pax_activity(self, route, STATIC_DWELL, DYNAMIC_DWELL, time_now, is_flex=is_flex)

        self.event['last']['time'] = self.event['next']['time']
        self.event['last']['type'] = 'arrive'
        self.event['last']['stop'] = self.event['next']['stop']

        if self.event['next']['stop'] == N_STOPS - 1:
            ## continue onto layover
            self.layover(route, dwell_time, time_now)
        else:
            self.event['next']['time'] = time_now + dwell_time
            self.event['next']['type'] = 'depart'

        ## update route
        stop_idx = self.event['next']['stop']
        route.stops[self.direction][stop_idx].last_arrival_time.append(time_now)
    
    def depart_stop(self, skip_flex=True):
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

    def get_latest_delay(self):
        return self.event_hist['arrival_time'][-1] - self.event_hist['scheduled_time'][-1]


class EventManager:
    def __init__(self) -> None:
        start_time = 0
        self.timestamps = [start_time]

        self.done = 0
        self.requires_control = 0
        self.veh_idx = None
        self.state_hist = {'time': [],'observation': [], 'action': [], 'tot_reward': [],
                           'reward_1': [], 'reward_2': [], 'reward_3': []}
    
    def start_vehicles(self, route):
        for vehicle in route.vehicles:
            vehicle.start(route)

    def step(self, route, action=None):
        if (action is not None) and (self.veh_idx is not None):
            ## reset counters
            for ky in route.inter_event:
                route.inter_event[ky] = 0

            ## update
            self.state_hist['action'].append(action)
            ## perform event
            route.vehicles[self.veh_idx].depart_stop(skip_flex=action)
            return self.step(route)
        
        self.veh_idx = find_closest_vehicle(route.vehicles, self.timestamps[-1])

        ## add new time to list of timestamps
        self.timestamps.append(route.vehicles[self.veh_idx].event['next']['time'])

        time_now = self.timestamps[-1]
        
        if time_now > MAX_TIME_HOURS*60*60:
            ## RETURN TUPLE ACCORDING TO GYM (OBSERVATION, REWARD, TERMINATED, TRUNCATED, INFO)
            obs = []
            reward = None
            terminated, truncated = 1, 1
            info = {}
            return obs, reward, terminated, truncated, info
        
        ## get all passengers up to the current time
        route.get_active_pax(time_now)

        requires_control = check_control_conditions(route.vehicles[self.veh_idx], CONTROL_STOPS)

        if requires_control:
            stop_idx = route.vehicles[self.veh_idx].event['next']['stop']
            direction = route.vehicles[self.veh_idx].direction
            
            flex_stop_idx = stop_idx + 1
            direction = route.vehicles[self.veh_idx].direction
            flex_stop = route.stops[direction][flex_stop_idx]
            n_flex_pax = len(flex_stop.active_pax)

            ## we will only request an action if there are flex route passengers waiting
            if n_flex_pax:
                ## TODO: state trimming and introudce condition if enough buffer time has passed since start of episode
                headway = int(route.stops[direction][stop_idx].get_latest_headway()/60)
                load = len(route.vehicles[self.veh_idx].pax)
                ## get the floor integer of the delay in minutes
                delay = round(route.vehicles[self.veh_idx].get_latest_delay()/60)
                ## RETURN TUPLE ACCORDING TO GYM (OBSERVATION, REWARD, TERMINATED, TRUNCATED, INFO)
                obs = [stop_idx, n_flex_pax, headway, load, delay]
                reward = route.get_reward(self)
                terminated, truncated = 0, 0
                info = route.inter_event

                self.state_hist['observation'].append(obs)
                self.state_hist['time'].append(time_now)
                return obs, reward, terminated, truncated, info
            
        ## if no control required 
        if route.vehicles[self.veh_idx].event['next']['type'] == 'arrive':
            route.vehicles[self.veh_idx].arrive_at_stop(route)
            return self.step(route)
        
        if route.vehicles[self.veh_idx].event['next']['type'] == 'depart':
            route.vehicles[self.veh_idx].depart_stop()
            return self.step(route)

