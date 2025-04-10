PATH_ROUTES = 'data/routes.csv'
PATH_STOPS = 'data/stops.csv'

## simulation parameters
MAX_TIME_HOURS = 4.0 # hours
BUFFER_SECONDS = 3600 # seconds
N_VEHICLES = 2
RESULTS_START_TIME_MINUTES = 30
RESULTS_END_TIME_MINUTES = 180

## route layout
N_STOPS = 7 # per direction
FLEX_STOPS = [2, 4]
CONTROL_STOPS = [i-1 for i in FLEX_STOPS]

## travel/dwell times
STATIC_DWELL = 6 # seconds
DYNAMIC_DWELL = 2 # seconds
SEGMENT_TIMES = {
    'flex': {'mean': 80, 'std': 20},
    'fixed': {'mean': 110, 'std': 30}
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

# define a conversion from (direction,stop) to control stop index
CONTROL_STOP_CONVERSION = {
    ('out', 1): 0,
    ('out', 3): 1,
    ('in', 1): 2,
    ('in', 3): 3
}

## additional parameters

## the amount of time a flex passenger is willing to wait
MAX_WAIT_TIME_FLEX = 15 # minutes
## boolean indicates whether to remove passengers waiting more than MAX_WAIT_TIME_FLEX
REMOVE_LONG_WAIT_FLEX = True

## time until a trip is considered late
SCHEDULE_TOLERANCE = 3 ## minutes
ON_TIME_BOUNDS = (-60, 60) ## seconds

## State definition
# Define the bounds for each state variable in the environment

REWARD_WEIGHTS = {
    'lost_requests': -1.0,
    'off_schedule_trips': -1.0
}

## SELECTIVE DEVIATION PARAMETERS
DELAY_THRESHOLD = 60

# DYNAMIC SMART GREEDY PARAMETERS
BASE_MINIMUM_REQUEST = 1
SLOPE_MINIMUM_REQUESTS = 1.0

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'