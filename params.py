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
STATIC_DWELL = 7 # seconds
DYNAMIC_DWELL = 2 # seconds
SEGMENT_TIMES = {
    'flex': {'mean': 80, 'std': 20},
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
SCHEDULE_TOLERANCE = 3 ## minutes
ON_TIME_BOUNDS = (-60, 120) ## seconds

## State definition
# Define the bounds for each state variable in the environment

REWARD_WEIGHTS = {
    'denied': 1.0,
    'early': 2.0,
    'late': 2.0
}

## SELECTIVE DEVIATION PARAMETERS
DELAY_THRESHOLD = 60

## DYNAMIC SMART GREEDY PARAMETERS
## define list where each item is the (max_delay, pax_threshold)
MIN_PAX_THRESHOLDS = [(0, 1), (120, 2), (240, 4) ,(1000, 5)]

## PATHS FOR OUTPUT
OUTPUT_FOLDER_PATH = 'outputs/'

# PARAM_BOUNDS = {
#     'stop_idx': {
#         'bins': [3],
#         'max_value': 3
#     },  
    
#     'n_flex_pax': {
#         'bins': [0, 1, 2],  # Bins: [0, 1, 2+]
#         'max_value': 2      # 2+ bin includes any value 2 or above
#     },
    
#     'headway': {
#         'bins': [5, 10],    # Bins: [0-5, 5-10, 10+]
#         'max_value': 10     # 10+ bin includes any value greater than 10
#     },
    
#     'load': {
#         'bins': [3],        # Bins: [0-3, 3+]
#         'max_value': 2      # 3+ bin includes any value greater than 2
#     },
    
#     'delay': {
#         'bins': [1, 2, 3],     # Bins: [<1, 1-3, 3+]
#         'max_value': 3      # 3+ bin includes any value greater than 3
#     }
# }

# def get_bin_index(actual_value, bounds):
#     # If bounds is a list, it's directly indexed (for stop_idx)
#     if isinstance(bounds, list):
#         return min(max(0, actual_value), bounds[1])

#     # For others, the bounds are defined as bins with maximum values
#     for i, bound in enumerate(bounds['bins']):
#         if actual_value <= bound:
#             return i

#     # If the value exceeds all bounds, return the last bin
#     return len(bounds['bins'])


# def get_binned_state(actual_state: list):
#     """
#     Maps the actual values in the state 

#     Parameters:
#     - actual_state: The list of actual values of the state variable

#     Returns:
#     - binned_state: The list of bin indices for each state variable
#     """

#     binned_state = []
#     for i, state_name in enumerate(STATE_KEYS):
#         bounds = PARAM_BOUNDS[state_name]
#         binned_state.append(get_bin_index(actual_state[i], bounds))
#     return binned_state