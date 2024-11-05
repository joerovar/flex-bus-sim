PATH_ROUTES = 'data/routes.csv'
PATH_STOPS = 'data/stops.csv'

## simulation parameters
MAX_TIME_HOURS = 4.0 # hours
BUFFER_SECONDS = 3600 # seconds
N_VEHICLES = 2

## route layout
N_STOPS = 7
FLEX_STOPS = [2, 4]
CONTROL_STOPS = [i-1 for i in FLEX_STOPS]

## travel/dwell times
STATIC_DWELL = 7 # seconds
DYNAMIC_DWELL = 2 # seconds
SEGMENT_TIMES = {
    'flex': {'mean': 85, 'std': 20},
    'fixed': {'mean': 125, 'std': 30}
}

## service design
SCHEDULE_HEADWAY = 10 ## minutes
N_TRIPS = 100
HALF_CYCLE_TIME = 10 ## minutes
SCHEDULED_STOP_TIMES = [
    0, 
    150, 
    225, ## flex - does not really matter
    300, 
    375, ## flex - does not really matter
    450,
    600]

## demand parameters
OD_LOW = 3.5
OD_HIGH = 6.5

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
SCHEDULE_TOLERANCE = 3 ## minutes
ON_TIME_BOUNDS = (-60, 120) ## seconds

## State definition
# Define the bounds for each state variable in the environment

REWARD_WEIGHTS = {
    'denied': 1.0,
    'early': 1.0,
    'late': 3.0
}

## SELECTIVE DEVIATION PARAMETERS
DELAY_THRESHOLD = 60

## DYNAMIC SMART GREEDY PARAMETERS
## define list where each item is the (max_delay, pax_threshold)
DEFAULT_MIN_PAX_THRESHOLDS = [(0, 1), (90, 2), (180, 4) ,(1000, 5)]

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'