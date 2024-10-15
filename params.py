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
# Define the bounds for each state variable in the environment

PARAM_BOUNDS = {
    'stop_idx': [0, 6],  # Bins: [0-6]
    
    'n_flex_pax': {
        'bins': [0, 1, 2],  # Bins: [0, 1, 2+]
        'max_value': 2      # 2+ bin includes any value 2 or above
    },
    
    'headway': {
        'bins': [5, 10],    # Bins: [0-5, 5-10, 10+]
        'max_value': 10     # 10+ bin includes any value greater than 10
    },
    
    'load': {
        'bins': [2],        # Bins: [0-2, 3+]
        'max_value': 2      # 3+ bin includes any value greater than 2
    },
    
    'delay': {
        'bins': [1, 3],     # Bins: [<1, 1-3, 3+]
        'max_value': 3      # 3+ bin includes any value greater than 3
    }
}

REWARD_WEIGHTS = {
    'denied': 1,
    'fixed_wait_time': 1,
    'late': 1
}



## SMART GREEDY PARAMETERS
SG_MAX_DELAY = 1

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'