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
    
    def get_latest_headway(self, time_now=None):
        if time_now is not None:
            return time_now - self.last_arrival_time[-1]
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
        self.lost_requests = []
        self.inter_event = [{'skipped_requests': 0, 'off_schedule_trips': 0} for _ in range(N_VEHICLES)]

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
                    lost_requests = self.stops[direction][i].remove_long_wait_pax(time_now)
                    self.lost_requests += lost_requests
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
        self.tracker = {
            'deviation_opportunities': 0,
            'deviations': 0,
            'requests_picked': [],
            'early_trips': 0,
            'late_trips': 0,
        }
    
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
        scheduled_time = route.get_scheduled_time(self.event['next']['stop'], self.trip_idx, self.direction)
        self.event_hist['scheduled_time'].append(scheduled_time)
        
        ## process
        is_flex = self.event['next']['stop'] in FLEX_STOPS
        dwell_time = pax_activity(self, route, STATIC_DWELL, DYNAMIC_DWELL, time_now, is_flex=is_flex)

        ## update the on-time arrivals tracker if it is a control stop or the terminal
        if self.event['next']['stop'] in [CONTROL_STOPS[1], N_STOPS-1]:
            schedule_deviation = time_now - scheduled_time
            early = schedule_deviation < ON_TIME_BOUNDS[0]
            late = schedule_deviation > ON_TIME_BOUNDS[1]
            if early:
                self.tracker['early_trips'] += 1
            elif late:
                self.tracker['late_trips'] += 1
            # if early or late:
            #     # print(f"Vehicle {self.idx}")
            #     # print(f"Current time: {time_now}, ")
            #     # print(f"Stop: {self.event['next']['stop']}, ")
            #     # print(f"Scheduled time: {scheduled_time}, Deviation: {schedule_deviation}")
            #     # print(f"----------")
            #     route.inter_event[self.idx]['off_schedule_trips'] += 1

        self.event['last']['time'] = self.event['next']['time']
        self.event['last']['type'] = 'arrive'
        self.event['last']['stop'] = self.event['next']['stop']

        ## update route
        stop_idx = self.event['next']['stop']
        route.stops[self.direction][stop_idx].last_arrival_time.append(time_now)
        
        if self.event['next']['stop'] == N_STOPS - 1:
            ## continue onto layover
            self.layover(route, dwell_time, time_now)
        else:
            self.event['next']['time'] = time_now + dwell_time
            self.event['next']['type'] = 'depart'

        # print(f"Vehicle {self.idx} arrived at stop {stop_idx} in direction {self.direction} at {time_now}")
    
    def depart_stop(self, route, deviate=0):
        time_now = self.event['next']['time']

        ## set destination
        self.event['last']['time'] = time_now
        self.event['last']['type'] = 'depart'
        self.event['last']['stop'] = self.event['next']['stop']
        self.event['next']['type'] = 'arrive'

        stop = self.event['next']['stop']

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

    def get_latest_schedule_deviation(self):
        return self.event_hist['arrival_time'][-1] - self.event_hist['scheduled_time'][-1]
    
    def get_location(self):
        return self.direction, self.event['next']['stop']


class EnvironmentManager:
    def __init__(self, reward_weight=TRIP_WEIGHT) -> None:
        start_time = 0
        self.timestamps = [start_time]

        self.done = 0
        self.requires_control = 0
        self.veh_idx = None
        self.unweighted_rewards_per_vehicle = {i: [] for i in range(N_VEHICLES)}
        self.state_hist = [{'time': [], 'observation': [], 'action': [], 
                           'reward': [], 'unweighted_rewards': [], 'next_observation': [], 'done': []} for i in range(N_VEHICLES)]
        self.route = RouteManager()
        self.reward_weight = reward_weight
    
    def start_vehicles(self):
        for vehicle in self.route.vehicles:
            vehicle.start(self.route)
    
    def get_history(self):
        history = {}
        history['pax'] = get_pax_history(self.route, FLEX_STOPS, include_denied=True)
        history['vehicles'] = get_vehicle_history(self.route.vehicles, FLEX_STOPS)
        
        # state_histories = []
        # for i in range(N_VEHICLES):
        #     if len(self.state_hist[i]['observation']) != len(self.state_hist[i]['reward']):
        #         # print(f"Lengths do not match for vehicle {i}: obs {len(self.state_hist[i]['observation'])} vs rew {len(self.state_hist[i]['reward'])}")
        #         self.state_hist[i]['time'].pop(-1)
        #         self.state_hist[i]['observation'].pop(-1)
        #     state_hist = pd.DataFrame(self.state_hist[i])
        #     state_hist['veh_idx'] = i
        #     state_histories.append(state_hist)
        
        # history['state'] = pd.concat(state_histories, ignore_index=True)
        
        history['idle'] = pd.DataFrame(self.route.idle_time)
        return history
    
    def get_tracker_info(self):
        # get deviation opportunities and deviations
        deviation_opp = 0
        deviations = 0
        requests_picked = []
        early_trips = 0
        late_trips = 0
        for vehicle in self.route.vehicles:
            deviation_opp += vehicle.tracker['deviation_opportunities']
            deviations += vehicle.tracker['deviations']
            requests_picked += vehicle.tracker['requests_picked']
            early_trips += vehicle.tracker['early_trips']
            late_trips += vehicle.tracker['late_trips']
        # get average of requests picked
        avg_requests_picked = np.mean(requests_picked) if len(requests_picked) > 0 else 0
        return deviation_opp, deviations, avg_requests_picked, early_trips, late_trips
        
    def step(self, action=None):
        if action is not None:
            # record in inter_event_count
            n_requests = self.state_hist[self.veh_idx]['observation'][-1][1]
            if n_requests > 0:
                self.route.vehicles[self.veh_idx].tracker['deviation_opportunities'] += 1
            # if len(self.unweighted_rewards_per_vehicle[self.veh_idx]) > 0:
            #     print(f"Vehicle {self.veh_idx} unweighted rewards: {self.unweighted_rewards_per_vehicle[self.veh_idx]}")
            #     print(f"Observation history: {self.state_hist[self.veh_idx]['observation']}")
            #     print(f"Next Observation history: {self.state_hist[self.veh_idx]['next_observation']}")
            #     print(f"Done history: {self.state_hist[self.veh_idx]['done']}")
            #     print(f"Action history: {self.state_hist[self.veh_idx]['action']}")
            #     print(f"-------")
            if action == 0:
                self.unweighted_rewards_per_vehicle[self.veh_idx].append(n_requests)
            else:
                self.route.vehicles[self.veh_idx].tracker['deviations'] += 1
                self.route.vehicles[self.veh_idx].tracker['requests_picked'].append(n_requests)
                self.unweighted_rewards_per_vehicle[self.veh_idx].append(0)
            # print(f"Vehicle {self.veh_idx} processed action {action} at stop {self.route.vehicles[self.veh_idx].event['next']['stop']} ")
            # update
            self.state_hist[self.veh_idx]['action'].append(action)
            # if len(self.timestamps) > 0 and self.veh_idx == 0:
            #     print(f"Added action {self.state_hist[self.veh_idx]['action'][-1]} for vehicle {self.veh_idx} at time {self.timestamps[-1]}")

            # perform action
            self.route.vehicles[self.veh_idx].depart_stop(self.route, deviate=action)
        
        self.veh_idx = find_next_event_vehicle_index(self.route.vehicles, self.timestamps[-1])

        ## add new time to list of timestamps
        self.timestamps.append(self.route.vehicles[self.veh_idx].event['next']['time'])
        time_now = self.timestamps[-1]

        ## get all passengers up to the current time
        self.route.get_active_pax(time_now)

        ## check control
        observation = None 
        done = 0
        # first, where is the vehicle
        vehicle = self.route.vehicles[self.veh_idx]
        direction, stop_idx = vehicle.get_location()
        stop = self.route.stops[direction][stop_idx]
        # if self.veh_idx == 1:
        #     print(f"---------")
        #     print(f"Vehicle {self.veh_idx} at stop {stop_idx} in direction {direction} at time {time_now}")
        # skip scenario 1: if stop not in control stops then observation is None
        if stop_idx not in CONTROL_STOPS:
            # if self.veh_idx == 1:
            #     print(f"Not in a control stop")
            observation = None
        elif stop_idx == CONTROL_STOPS[-1] and len(stop.last_arrival_time) > 0:
            # if self.veh_idx == 1:
            #     print(f"processed as terminal")
            # if next stop is last stop, then make terminal observation
            observation = [
                np.int32(STOP_TO_CONTROL_STOP_MAP[stop_idx]), 
                0,
                np.float32(stop.get_latest_headway(time_now=time_now)), 
                np.float32(vehicle.get_latest_schedule_deviation())
            ]
            done = 1
        elif vehicle.event['next']['type'] == 'depart' and len(stop.last_arrival_time) > 1:
            # if self.veh_idx == 1:
                # print(f"processed as regular departure")
            # if trip is departing, then make observation
            flex_stop_idx = stop_idx + 1
            n_requests = self.route.get_n_waiting_pax(flex_stop_idx, direction)
            observation = [
                np.int32(STOP_TO_CONTROL_STOP_MAP[stop_idx]), 
                np.int32(n_requests),
                np.float32(stop.get_latest_headway()), 
                np.float32(vehicle.get_latest_schedule_deviation())
            ]
            done = 0

        if observation is not None:
            ## bookkeeping
            if not done:
                self.state_hist[self.veh_idx]['observation'].append(observation)
                self.state_hist[self.veh_idx]['time'].append(time_now)
            # if self.veh_idx == 0:
            #     print(f"Added observations {self.state_hist[self.veh_idx]['observation'][-1]} for vehicle {self.veh_idx} at time {time_now}")
            if stop_idx != CONTROL_STOPS[0]:
                self.state_hist[self.veh_idx]['done'].append(done)
                self.state_hist[self.veh_idx]['next_observation'].append(observation)
            # let's print out the length for everything if it is veh_idx=0

            # if the headway is less than a threshold, we set the next time as the difference
            headway_threshold = 150 # statistically derived
            headway = observation[2]
            diff_headway = headway - headway_threshold
            if diff_headway < 0:
                # advance the clock
                self.route.vehicles[self.veh_idx].event['next']['time'] += abs(diff_headway)
                
                # check if time has exceeded
                if time_now > MAX_TIME_HOURS*60*60 - BUFFER_SECONDS:
                    terminated = 1
                    reward = 0
                    info = {}
                    return observation, reward, done, terminated, info
                
                # continue to next step
                return self.step(action=None)
            
            # get reward
            if stop_idx == CONTROL_STOPS[0] or (len(self.state_hist[self.veh_idx]['observation'])==0):
                # scenario 1: if stop is the first of control stops or if there are no previous observations
                # then the rewards are null because this is first state
                reward = np.nan
            else:
                unweighted_rewards = self.unweighted_rewards_per_vehicle[self.veh_idx]
                delay = observation[3]
                off_schedule = (delay > ON_TIME_BOUNDS[1]) or (delay < ON_TIME_BOUNDS[0])
                if off_schedule:
                    # scenario 2: if the vehicle is off schedule, then the reward is zero
                    unweighted_rewards.append(1)
                else:
                    # scenario 3: if the vehicle is on schedule, then the reward is the sum of the unweighted rewards
                    unweighted_rewards.append(0)
                # if len(unweighted_rewards) < 2:
                #     print(f"Vehicle {self.veh_idx} unweighted rewards: {self.state_hist[self.veh_idx]['unweighted_rewards']}")
                #     print(f"stop index: {stop_idx}")
                #     print(f"Observation history: {self.state_hist[self.veh_idx]['observation']}")
                #     print(f"Next Observation history: {self.state_hist[self.veh_idx]['next_observation']}")
                #     print(f"Action history: {self.state_hist[self.veh_idx]['action']}")
                #     print(f"current weighted rewards: {unweighted_rewards}")
                #     print(f"Done history: {self.state_hist[self.veh_idx]['done']}")
                weighted_rewards = [-1.0 *unweighted_rewards[0], TRIP_WEIGHT * unweighted_rewards[1]]
                reward = np.float32(np.sum(weighted_rewards))
                self.state_hist[self.veh_idx]['unweighted_rewards'].append(unweighted_rewards)
                self.state_hist[self.veh_idx]['reward'].append(reward)
                
                # reset unweighted rewards
                self.unweighted_rewards_per_vehicle[self.veh_idx] = []
            
            # if self.veh_idx == 0:
            #     for ky in self.state_hist[self.veh_idx]:
            #         print(f"Time: {time_now}")
            #         print(f"Length of {ky}: {len(self.state_hist[self.veh_idx][ky])}")
            #     print("-------")

            # if self.state_hist[self.veh_idx]['observation']:
            #     inter_event_counts = self.route.inter_event[self.veh_idx]
            #     # if self.veh_idx == 0:
            #     #     print(f"Vehicle {self.veh_idx} inter_event_counts: {inter_event_counts['skipped_requests']}")
            #     reward, unweighted_rewards = get_reward(inter_event_counts, self.reward_weight)
            #     self.state_hist[self.veh_idx]['reward'].append(reward)
            #     self.state_hist[self.veh_idx]['unweighted_rewards'].append(unweighted_rewards)
            #     # if self.veh_idx == 0:
            #     #     print(f"Added rewards {self.state_hist[self.veh_idx]['unweighted_rewards'][-1]} for vehicle {self.veh_idx} at time {time_now}")
            # else:
            #     reward, unweighted_rewards = np.nan, np.nan

            info = self.route.inter_event[self.veh_idx].copy()
            ## add time
            info['time'] = time_now
            ## vehicle index
            info['veh_idx'] = self.veh_idx
            info['direction'] = self.route.vehicles[self.veh_idx].direction

            if time_now > MAX_TIME_HOURS*60*60 - BUFFER_SECONDS:
                terminated = 1
                return observation, reward, done, terminated, info
            
            if done:
                self.route.vehicles[self.veh_idx].arrive_at_stop(self.route)
            terminated = 0
            return observation, reward, done, terminated, info
                
        ## if no control required 
        if self.route.vehicles[self.veh_idx].event['next']['type'] == 'arrive':
            self.route.vehicles[self.veh_idx].arrive_at_stop(self.route)
            return self.step()
        
        if self.route.vehicles[self.veh_idx].event['next']['type'] == 'depart':
            self.route.vehicles[self.veh_idx].depart_stop(self.route)
            return self.step()

