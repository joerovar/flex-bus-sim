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
    'flex': {'mean': 90, 'std': 15},
    'fixed': {'mean': 120, 'std': 30}
}

## service design
SCHEDULE_HEADWAY = 10 ## minutes
N_TRIPS = 100
HALF_CYCLE_TIME = 10 ## minutes

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


REWARD_WEIGHTS = {
    'denied': 1,
    'fixed_wait_time': 1,
    'late': 1
}

## SMART GREEDY PARAMETERS
SG_MAX_DELAY = 1

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'
