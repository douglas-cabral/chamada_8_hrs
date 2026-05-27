# Sample script on how to use the thrust_matching function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
from designTool.aerodynamics import aerodynamics
from designTool.auxiliary import atmosphere
from designTool.constants import gravity
import pprint

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# Execute the geometry function (S_w vem de standard_airplane, ponto de projeto)
geometry(airplane)

inp = airplane['inputs']
W0_guess = inp['W0_guess']
T0_guess = inp['n_engines'] * inp['engine']['Tmax']

thrust_matching(W0_guess, T0_guess, airplane)

# Print results
print("airplane['thrust_matching'] = " + pprint.pformat(airplane['thrust_matching']))

# --- Ponto de projeto: CLmax e distâncias (S_w, T0, W0 do dicionário / thrust_matching) ---
tm = airplane['thrust_matching']
W0 = tm['W0']
T0 = tm['T0']
S_w = inp['S_w']
CLmaxTO = tm['CLmaxTO']

_, CLmaxLD, _ = aerodynamics(
    airplane,
    Mach=0.2,
    altitude=inp['altitude_landing'],
    CL=0.5,
    n_engines_failed=0,
    highlift_config='landing',
    lg_down=1,
    h_ground=inp['h_ground'],
)

atm_to = atmosphere(inp['altitude_takeoff'], inp['deltaISA_takeoff'])
sigma = atm_to['density'] / 1.225
T0W0 = T0 / W0
distance_takeoff_avail = 0.2387 / (sigma * CLmaxTO * T0W0) * (W0 / S_w)

atm_ld = atmosphere(inp['altitude_landing'], inp['deltaISA_landing'])
rho_ld = atm_ld['density']
h_land = 15.3
f_land = 5 / 3
a_g = 0.5
x_land = 1.52 / a_g + 1.69
A_land = gravity / f_land / x_land
B_land = -10.0 * gravity * h_land / x_land
MLW_frac = inp['MLW_frac']

distance_landing_avail = (
    W0 * MLW_frac / (rho_ld * S_w * CLmaxLD) - B_land
) / A_land

print("\n--- Ponto de projeto (S_w em standard_airplane, T0 = Tmax·n_eng) ---")
print(f"  gravity          = {gravity} m/s²")
print(f"  S_w              = {S_w:.2f} m²")
print(f"  T0               = {T0/1000:.1f} kN  ({inp['n_engines']} motores)")
if 'Tmax' in inp['engine']:
    print(f"  Tmax (motor)     = {inp['engine']['Tmax']/1000:.1f} kN")
print(f"  W0 (MTOW)        = {W0/gravity:.0f} kg")
print(f"  CLmaxTO          = {CLmaxTO:.4f}")
print(f"  CLmaxLD          = {CLmaxLD:.4f}")
print(f"  Dist. decolagem (compatível com T0, W0, S_w) = {distance_takeoff_avail:.1f} m")
print(f"  Dist. pouso (Torenbeek)                      = {distance_landing_avail:.1f} m")
print(f"  Requisitos no inputs: d_TO = {inp['distance_takeoff']:.1f} m, "
      f"d_LD = {inp['distance_landing']:.1f} m")
print(f"  Margem pouso (deltaS_wlan)                   = {tm['deltaS_wlan']:.2f} m²")
