PATH_ROUTES = 'data/routes.csv'
PATH_STOPS = 'data/stops.csv'

N_STOPS = 7
FLEX_STOPS = [2, 4]
CONTROL_STOPS = [i-1 for i in FLEX_STOPS]
MAX_TIME_HOURS = 3 # hours
N_VEHICLES = 2
STATIC_DWELL = 5 # seconds
DYNAMIC_DWELL = 2 # seconds
SEGMENT_TIMES = {
    'flex': {'mean': 90, 'std': 30},
    'fixed': {'mean': 120, 'std': 60}
}

ARRIVAL_RATES = {
            'flex': 6,
            'fixed': 10,
            'terminal': 16
}

MAX_WAIT_TIME_FLEX = 15 # minutes
REMOVE_LONG_WAIT_FLEX = True
SCHEDULE_TOLERANCE = 3

## State definition
STATE_DIM = 3

LOW = [
    0, ## HEADWAY
    0, ## Occupancy
    0 ## Control stop
]

HIGH = [
    3, # headway
    3, # occupancy
    2, # control stop
]

