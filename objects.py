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
        self.last_arrival_time = []
        ## only for control stops
        self.last_action = 0
    
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
        # print(self.last_arrival_time)
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
            'fixed_boardings': 0,
            'early_trips': 0,
            'late_trips': 0,
            'total_trips': 0
        }

    def load_all_pax(self):
        for direction in self.stops:
            set_stops = self.stops[direction]
            for i in range(len(set_stops)-1):
                inter_arrival_times = []
                for j in range(len(set_stops)):
                    od_rate = OD_MATRIX[i][j]
                    if od_rate > 0:
                        if direction == 'out':
                            arrival_times = list(get_poisson_arrival_times(od_rate, MAX_TIME_HOURS))
                        else:
                            arrival_times = list(get_poisson_arrival_times(od_rate, MAX_TIME_HOURS, start_time=HALF_CYCLE_TIME*60))
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

    def get_excess_wait_time(self):
        avg_wait_time = self.inter_event['fixed_wait_time'] / np.max([self.inter_event['fixed_boardings'], 1]) 
        excess_wait_time = (avg_wait_time/60) - (SCHEDULE_HEADWAY/2)
        excess_wait_time = max(0, excess_wait_time)
        return excess_wait_time

    def get_reward(self, event):
        ## reward 1: denied passengers
        denied_pax = self.inter_event['denied']
        reward_denied = -1 * denied_pax * REWARD_WEIGHTS['denied']
        
        ## reward 2: fixed wait time
        # excess_wait_time = self.get_excess_wait_time()
        # reward_2 = excess_wait_time * REWARD_WEIGHTS['fixed_wait_time']
        
        early_trips = self.inter_event['early_trips']
        reward_early_trips = -1 * early_trips * REWARD_WEIGHTS['early']
        
        late_trips = self.inter_event['late_trips']
        reward_late_trips = -1 * late_trips * REWARD_WEIGHTS['late']

        unweighted_rewards = [denied_pax, early_trips, late_trips]
        event.state_hist['unweighted_rewards'].append(unweighted_rewards)

        reward = np.float32(reward_denied + reward_early_trips + reward_late_trips)
        event.state_hist['reward'].append(reward)
        return reward
    
    def get_n_waiting_pax(self, stop_idx, direction):
        return len(self.stops[direction][stop_idx].active_pax)

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

        ## update the on-time arrivals counter
        delay = self.event['next']['time'] - next_schd_time
        if delay > ON_TIME_BOUNDS[1]:
            route.inter_event['late_trips'] += 1
        if delay < ON_TIME_BOUNDS[0]:
            route.inter_event['early_trips'] += 1

        ## update the total trips
        route.inter_event['total_trips'] += 1
        
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
        # print(f"Vehicle {self.idx} arrived at stop {stop_idx} in direction {self.direction} at {time_now}")
    
    def depart_stop(self, route, deviate=0):
        time_now = self.event['next']['time']

        ## set destination
        self.event['last']['time'] = time_now
        self.event['last']['type'] = 'depart'
        self.event['last']['stop'] = self.event['next']['stop']
        self.event['next']['type'] = 'arrive'

        stop = self.event['next']['stop']

        ## record action
        route.stops[self.direction][self.event['next']['stop']].last_action = deviate

        if (not deviate) and (stop in CONTROL_STOPS):
            ## advance to next fixed stop (two stops ahead)
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
    
    def get_location(self):
        return self.event['next']['stop'], self.direction


class EnvironmentManager:
    def __init__(self, agents='independent') -> None:
        start_time = 0
        self.timestamps = [start_time]

        self.done = 0
        self.requires_control = 0
        self.veh_idx = None
        self.state_hist = {'time': [],'observation': [], 'action': [], 'reward': [], 'unweighted_rewards': []}
        self.agents = agents
        self.route = RouteManager()
    
    def start_vehicles(self):
        for vehicle in self.route.vehicles:
            vehicle.start(self.route)

    def step(self, action=None):
        if action is not None:
            ## update
            self.state_hist['action'].append(action)
            ## perform event
            self.route.vehicles[self.veh_idx].depart_stop(self.route, deviate=action)
        
        self.veh_idx = find_next_event_vehicle_index(self.route.vehicles, self.timestamps[-1])

        ## add new time to list of timestamps
        self.timestamps.append(self.route.vehicles[self.veh_idx].event['next']['time'])
        time_now = self.timestamps[-1]
    
        if time_now > MAX_TIME_HOURS*60*60:
            print('crazy')
            print(time_now)
            raise ValueError(f'Time limit exceeded at {time_now}')

        ## get all passengers up to the current time
        self.route.get_active_pax(time_now)

        ## check control
        observation = get_observation(
            self.route.vehicles[self.veh_idx], self.route, CONTROL_STOPS)

        if observation is not None:           
            ## reward
            if self.state_hist['observation']:
                reward = self.route.get_reward(self)
            else:
                reward = np.nan

            info = self.route.inter_event.copy()
            ## add time
            info['time'] = time_now
            ## time passed since last event
            if self.state_hist['observation']:
                info['time_passed'] = time_now - self.state_hist['time'][-1]
                info['cumul_reward'] = np.sum(self.state_hist['reward'])
            else:
                info['time_passed'] = np.nan
                info['cumul_reward'] = np.nan
            ## vehicle index
            info['veh_idx'] = self.veh_idx
            info['direction'] = self.route.vehicles[self.veh_idx].direction
            ## reset counters after using it for reward and info
            for ky in self.route.inter_event:
                self.route.inter_event[ky] = 0

            if time_now > MAX_TIME_HOURS*60*60 - BUFFER_SECONDS:
                terminated, truncated = 1, 1
                return observation, reward, terminated, truncated, info
            ## bookkeeping
            self.state_hist['observation'].append(observation)
            self.state_hist['time'].append(time_now)
            terminated, truncated = 0, 0
            # vehicle_state_hist = route.vehicles[self.veh_idx].state_hist
            # if len(vehicle_state_hist['observation']):
            #     ## update the reward for the vehicle
            #     pass
            return observation, reward, terminated, truncated, info
                
        ## if no control required 
        if self.route.vehicles[self.veh_idx].event['next']['type'] == 'arrive':
            self.route.vehicles[self.veh_idx].arrive_at_stop(self.route)
            return self.step()
        
        if self.route.vehicles[self.veh_idx].event['next']['type'] == 'depart':
            self.route.vehicles[self.veh_idx].depart_stop(self.route)
            return self.step()

