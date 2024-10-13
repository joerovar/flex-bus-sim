# flex-bus-sim
Simulator for object oriented bus simulator with extensions for flexible route systems

## overall logic
This is an event-based simulator. Thus the simulation advances from one event (e.g., vehicle 1 arrives at stop X) to the immediate next event (e.g., vehicle 2 departs stop X). This is achieved by having each vehicle keep track of their own next event. At each simulation step:
- each vehicle's next event is stored in a list
- the closest event time (and the index of its corresponding vehicle, as well as the type of event) is selected and the information is used for processing next event

## files
- `main.py` has the experiments ready to run (neglect Gym framework for now, RL experiments incoming). Here you can modify for which experiments to run. 
- `params.py` includes the parameters for the experiments, modify this according to your network
- `objects.py` defines the necessary objects for the simulator. If you're just experimenting, you do not need to modify anything here.
- `helpers.py` has functions that the `objects.py` rely on 
- `utils.py` has function to be used for processing results and visualizations
