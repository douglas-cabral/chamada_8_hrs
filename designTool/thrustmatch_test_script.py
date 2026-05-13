# Sample script on how to use the thrust_matching function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('fokker100')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration
W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

# Execute the weight and thrust estimation
thrust_matching(W0_guess, T0_guess, airplane)

# Print results
print("airplane['thrust_matching'] = " + pprint.pformat(airplane['thrust_matching']))
