# IPTDX (InterPlanetary Trajectory Design toolboX)
This toolbox provides utility functions that helps the interplanetary trajectory design process. 
Python codes are to complement the [pykep](https://esa.github.io/pykep/index.html) library.
Some of the MATLAB Codes relies on external functions.

## Boundary Value Problems
A standard apporach to construct database of interplanetary transfers is to solve boundary value problems
with position and time epoch of the departure and arrival planet as boundary conditions. 


### 1. VILT-BVP
Solve for V-infinity leveraging transfer as a boundary value problem. The maneuver locations are constrained to
either the periapsis or the apoapsis. 
The boundary conditions are the time of flight and the position of departure and arrival.

Source Code:
- Python: vilt_bvp.py
- MATLAB: vilt_bvp.m  (plot: plot_VILT_PQW.m)

Demo Script: 
- Python: demo_vilt_bvp.ipynb
- MATLAB: demo_vilt_bvp.m

Reference: 

[1] Lantulh, D.V, Russel R.P., and Campagnola S. "Automated Inclution of V-Infinity Leverging Maneuvers in Gravity-Assist Flyby Tour Design", AAS 12-162


### 2. Optimal Maneuver Placement
This code solves for the optimal maneuver using the primer vector theory. 
The boundary conditions are the time of flight and the position of departure and arrival.
```The current implementation still has issues on the robustness of the flight time matching.```

Source Code:
- Python: vilt_bvp.py

Demo Script: 
- Python: demo_vilt_bvp.ipynb

Reference: 

[1] Landau, D. "Efficient Maneuver Placement for Automated Trajectory Design", JGCD, 2018

### 3. Partial Derivatives of the Lambert Problem
Partial derivatives of the lambert arcs could be used to interpolate the database using derivative information.
This code provides the partial derivatives of the initial and terminal velocity of the lambert arc with respect to
the position and time at both ends.

Source Code:
- Python: lambert_derivatives.py

Demo Script:
- Python: demo_lambert_derivatives.ipynb

Reference:

[1] Arora, N., Russel, R.P., Strange, N., and Ottesen, D. "Partial Derivatives of the Solution to the Lambert Boundary 
Value Problem", JGCD, 2015

## Visualizations

### 1. Tisserand Graph
Tisserand Graph could help construct a sequence of encounters between starting and designating orbits. 
This code generate Tisserand Graph of specified planets for a specified V-infinity range. 

Source Code:
- Python: tisserand_graph.py

Demo Script:
- Python: demo_tisserand_graph.ipynb

Reference:

[1] Sangra, D., Fantino, E., Flores, R., Lozano, O.C., and Estelrich, C.G. "An Automatic Tree Search Algorithm for the Tisserang Graph", 
Alexendria Enginnering Journal, 2015