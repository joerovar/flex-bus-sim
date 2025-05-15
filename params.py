PATH_ROUTES = 'data/routes.csv'
PATH_STOPS = 'data/stops.csv'

## simulation parameters
MAX_TIME_HOURS = 5.0 # hours
BUFFER_SECONDS = 3600 # seconds
N_VEHICLES = 2

## route layout
N_STOPS = 7 # per direction
FLEX_STOPS = [2, 4]
CONTROL_STOPS = [1, 3, 6]
STOP_TO_CONTROL_STOP_MAP = {
    1:0,
    3:1,
    6:2
}

CONTROL_STOPS_DIRECTION_MAPPING = {
    0: 'out',
    1: 'out',
    2: 'in',
    3: 'in'
}
CONTROL_STOPS_STOP_MAPPING = {
    0: 1,
    1: 3,
    2: 1,
    3: 3
}

## travel/dwell times
STATIC_DWELL = 8 # seconds
DYNAMIC_DWELL = 2.1 # seconds
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
def get_od_matrix(od_low, od_high):
    return [
        [0, od_low, 0, od_high, 0, od_high, 0],
        [0, 0, 0, od_low, 0, od_high, od_high],
        [0, 0, 0, 0, 0, od_low, od_low], ## FLEX
        [0, 0, 0, 0, 0, od_low, od_high],
        [0, 0, 0, 0, 0, 0, od_low], ## FLEX
        [0, 0, 0, 0, 0, 0, od_low]
    ]

# peak scenario
OD_LOW = 3.8
OD_HIGH = 7.5
OD_LOW_OFF_PEAK = 4.8
OD_HIGH_OFF_PEAK = 5.5

OD_MATRIX = {
    'peak': get_od_matrix(OD_LOW, OD_HIGH),
    'off_peak': get_od_matrix(OD_LOW_OFF_PEAK, OD_HIGH_OFF_PEAK)
}

## additional parameters

## the amount of time a flex passenger is willing to wait
MAX_WAIT_TIME_FLEX = 15 # minutes
## boolean indicates whether to remove passengers waiting more than MAX_WAIT_TIME_FLEX
REMOVE_LONG_WAIT_FLEX = True

## time until a trip is considered late
SCHEDULE_TOLERANCE = 3 ## minutes
ON_TIME_BOUNDS = (-90, 60) ## seconds

## State definition
# Define the bounds for each state variable in the environment

ALL_REWARD_WEIGHTS = {
    'skipped_requests': -1.0,
    'off_schedule_trips': -1.0
}
TRIP_WEIGHT = -4.0 # weight for the trip reward

## SELECTIVE DEVIATION PARAMETERS
DELAY_THRESHOLD = 60

# DYNAMIC SMART GREEDY PARAMETERS
# REQUESTS_MIN = ALPHA + BETA * (SCHEDULE_DEVIATION)
HEURISTIC_ALPHA = 1.0
HEURISTIC_BETA = 1.0
BASE_MINIMUM_REQUEST = 1
SLOPE_MINIMUM_REQUESTS = 1.0

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'

# FOR RL Training
STEPS_PER_EPISODE = 65 # based on testing