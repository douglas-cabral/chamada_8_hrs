# Sample script on how to use the balance function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
from designTool.balance import balance
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration
W0_guess = airplane['inputs']['W0_guess']
T0_guess = airplane['inputs']['n_engines']*airplane['inputs']['engine']['Tmax']

# Execute the weight and thrust estimation
thrust_matching(W0_guess, T0_guess, airplane)

# Execute the balance analysis
balance(airplane)

# Print results
print("airplane['balance'] = " + pprint.pformat(airplane['balance']))
