class Stop:
    def __init__(self, flex=False):
        self.pax = 0
        self.flex = flex

    def generate_pax(t):
        return


class Route:
    def __init__(self):
        n_stops = 7
        flex_stops = [3, 5]
        self.stops = {
            'in': [Stop(i in flex_stops) for i in range(n_stops)],
            'out': [Stop(i in flex_stops) for i in range(n_stops)]
        }

        demand = {
            'out': 75,
            'in': 60
        }
        
        ## in case they are flex segments

class Vehicle:
    def __init__(self, id, route, capacity):
        self.id = id
        self.route = route
        self.capacity = capacity
        self.location = route.sequence[0]

class EventManager:
    def __init__(self) -> None:
        
    


# class Passenger:
#     def __init__(self, id, source_stop, destination_stop, arrival_time):
#         self.id = id
#         self.source_stop = source_stop
#         self.destination_stop = destination_stop
#         self.arrival_time = arrival_time



# class Driver:
#     def __init__(self, id, vehicle):
#         self.id = id
#         self.vehicle = vehicle
#         self.status = 'idle'

