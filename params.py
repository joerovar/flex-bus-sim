PATH_ROUTES = 'data/routes.csv'
PATH_STOPS = 'data/stops.csv'

## simulation parameters
MAX_TIME_HOURS = 3 # hours
N_VEHICLES = 2

## route layout
N_STOPS = 7
FLEX_STOPS = [2, 4]
CONTROL_STOPS = [i-1 for i in FLEX_STOPS]

## travel/dwell times
STATIC_DWELL = 5 # seconds
DYNAMIC_DWELL = 2 # seconds
SEGMENT_TIMES = {
    'flex': {'mean': 75, 'std': 15},
    'fixed': {'mean': 120, 'std': 30}
}

## service design
SCHEDULE_HEADWAY = 10 ## minutes
N_TRIPS = 100
HALF_CYCLE_TIME = 10 ## minutes
SCHEDULED_LINK_TIME = 120 ## seconds
SCHEDULED_STOP_TIMES = [
    0, 
    150, 
    225, ## flex - does not really matter
    300, 
    375, ## flex - does not really matter
    450,
    600]

## demand parameters
OD_LOW = 3
OD_HIGH = 6

OD_MATRIX = [
    [0, OD_LOW, 0, OD_HIGH, 0, OD_HIGH, 0],
    [0, 0, 0, OD_LOW, 0, OD_LOW, OD_HIGH],
    [0, 0, 0, 0, 0, OD_LOW, OD_LOW], ## FLEX
    [0, 0, 0, 0, 0, OD_LOW, OD_HIGH],
    [0, 0, 0, 0, 0, 0, OD_LOW], ## FLEX
    [0, 0, 0, 0, 0, 0, OD_LOW]
]


## additional parameters

## the amount of time a flex passenger is willing to wait
MAX_WAIT_TIME_FLEX = 10 # minutes
## boolean indicates whether to remove passengers waiting more than MAX_WAIT_TIME_FLEX
REMOVE_LONG_WAIT_FLEX = True

## time until a trip is considered late
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
