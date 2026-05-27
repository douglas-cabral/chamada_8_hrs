# Sample script on how to use the balance function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
from designTool.balance import balance, tank_properties
from designTool.constants import gravity
import numpy as np

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration (ponto de projeto: S_w e Tmax do standard_airplane)
W0_guess = airplane['inputs']['W0_guess']
T0_guess = airplane['inputs']['n_engines'] * airplane['inputs']['engine']['Tmax']

# Execute the weight and thrust estimation
thrust_matching(W0_guess, T0_guess, airplane)

# Balance (neutral point and CG range)
balance(airplane)

geo = airplane['geometry']
cm_w = geo['cm_w']
xm_w = geo['xm_w']
xnp = airplane['balance']['xnp']

V_maxfuel, W_maxfuel, xcg_fuel, ycg_fuel = tank_properties(cr_w = airplane['geometry']['cr_w'],
                                                           ct_w = airplane['geometry']['ct_w'],
                                                           tcr_w = airplane['inputs']['tcr_w'],
                                                           tct_w = airplane['inputs']['tct_w'],
                                                           b_w = airplane['geometry']['b_w'],
                                                           sweep_w = airplane['inputs']['sweep_w'],
                                                           xr_w = airplane['inputs']['xr_w'],
                                                           x_tank_c_w = airplane['inputs']['x_tank_c_w'],
                                                           c_tank_c_w = airplane['inputs']['c_tank_c_w'],
                                                           b_tank_b_w_start = airplane['inputs']['b_tank_b_w_start'],
                                                           b_tank_b_w_end = airplane['inputs']['b_tank_b_w_end'],
                                                           rho_fuel = airplane['inputs']['rho_fuel'],
                                                           gravity = gravity)

tm = airplane['thrust_matching']
ew = airplane['empty_weight']

W_fuel = tm['W_fuel']
W_empty = tm['W_empty']
xcg_empty = ew['xcg_empty']

# Print results
print(f"cm_w: {cm_w} m")
print(f"xm_w: {xm_w} m")
print(f"xnp: {xnp} m")
print(f"xcg_fuel: {xcg_fuel} m")
print(f"W_fuel: {W_fuel} N  ({W_fuel / gravity} kg)")
print(f"W_empty: {W_empty} N  ({W_empty / gravity} kg)")
print(f"xcg_empty: {xcg_empty} m")
