# Sample script on how to use the full analysis function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.analyze import analyze
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# Execute the analysis function
analyze(airplane, print_log=True, plot=True)
