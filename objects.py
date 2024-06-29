class Stop:
    def __init__(self, id, name, stop_type, demand):
        self.id = id
        self.name = name
        self.stop_type = stop_type
        self.demand = demand

class Route:
    def __init__(self, id, name, stops, sequence):
        self.id = id
        self.name = name
        self.stops = stops
        self.sequence = sequence

class Passenger:
    def __init__(self, id, source_stop, destination_stop, arrival_time):
        self.id = id
        self.source_stop = source_stop
        self.destination_stop = destination_stop
        self.arrival_time = arrival_time

class Vehicle:
    def __init__(self, id, route, capacity):
        self.id = id
        self.route = route
        self.capacity = capacity
        self.location = route.sequence[0]

class Driver:
    def __init__(self, id, vehicle):
        self.id = id
        self.vehicle = vehicle
        self.status = 'idle'
