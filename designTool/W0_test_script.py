# Script de estimativa de peso W0 para a aeronave de projeto.

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.weight import weight
from designTool.constants import gravity, nm2m, ft2m
import numpy as np

def _v(x):
    """Representacao numerica sem arredondamento de exibicao."""
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    return repr(x)

# Load airplane
airplane = standard_airplane('my_airplane')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration
W0_guess = airplane['inputs']['W0_guess']
T0_guess = 862000.0

# Execute the weight estimation
W0, W_empty, W_fuel, W_cruise = weight(W0_guess, T0_guess, airplane)

# ============================================================
# 1. DESCRICAO DA MISSAO
# ============================================================
inp = airplane['inputs']

print("=" * 70)
print("  1. DESCRICAO DA MISSAO DE PROJETO")
print("=" * 70)
print()
print("  Tipo de aeronave       : Transporte de passageiros")
print("  Configuracao           : 320 assentos em 3 classes")
print("  Peso por passageiro    : 100 kg (incluindo bagagem)")
print("  Numero de motores      : %s" % _v(inp['n_engines']))
print()
print("  --- Missao de Projeto ---")
print("  Alcance de cruzeiro    : %s nmi" % _v(inp['range_cruise'] / nm2m))
print("  Mach de cruzeiro       : %s" % _v(inp['Mach_cruise']))
print("  Altitude de cruzeiro   : %s ft (%s m)" % (_v(inp['altitude_cruise'] / ft2m), _v(inp['altitude_cruise'])))
print("  Mach maximo operacional: %s" % _v(inp['Mach_maxcruise']))
print("  Altitude max cruzeiro  : %s ft (%s m)" % (_v(inp['altitude_maxcruise'] / ft2m), _v(inp['altitude_maxcruise'])))
print()
print("  --- Decolagem ---")
print("  Comprimento de pista   : %s m" % _v(inp['distance_takeoff']))
print("  Altitude               : %s m (nivel do mar)" % _v(inp['altitude_takeoff']))
print("  Delta ISA              : %s C (condicoes ISA)" % _v(inp['deltaISA_takeoff']))
print()
print("  --- Reservas ---")
print("  Tempo de espera (loiter): %s min" % _v(inp['time_loiter'] / 60))
print("  Altitude de loiter      : %s ft (%s m)" % (_v(inp['altitude_loiter'] / ft2m), _v(inp['altitude_loiter'])))
print("  Cruzeiro alternativo    : %s nmi" % _v(inp['range_altcruise'] / nm2m))
print("  Mach alternativo        : %s" % _v(inp['Mach_altcruise']))
print("  Altitude alternativa    : %s m (%s ft)" % (_v(inp['altitude_altcruise']), _v(inp['altitude_altcruise'] / ft2m)))
print()
print("  --- Pesos de Entrada ---")
print("  W_payload (320x100 kg) : %s N (%s kg)" % (_v(inp['W_payload']), _v(inp['W_payload'] / gravity)))
print("  W_crew                 : %s N (%s kg)" % (_v(inp['W_crew']), _v(inp['W_crew'] / gravity)))
print()

# ============================================================
# 2. BREAKDOWN DE PESO
# ============================================================
ew = airplane['empty_weight']

print("=" * 70)
print("  2. BREAKDOWN DE PESO DA AERONAVE")
print("=" * 70)
print()

components = [
    ("Asa (W_w)", ew['W_w']),
    ("Empenagem horizontal (W_h)", ew['W_h']),
    ("Empenagem vertical (W_v)", ew['W_v']),
    ("Fuselagem (W_f)", ew['W_f']),
    ("Trem de nariz (W_nlg)", ew['W_nlg']),
    ("Trem principal (W_mlg)", ew['W_mlg']),
    ("Motores instalados (W_eng)", ew['W_eng']),
    ("Demais sistemas (W_allelse)", ew['W_allelse']),
]

for name, val in components:
    print("  %s" % name)
    print("    Peso [N]   : %s" % _v(val))
    print("    Peso [kg]  : %s" % _v(val / gravity))
    print("    %% MTOW    : %s" % _v(val / W0 * 100))
    print()

print("  PESO VAZIO (W_empty)")
print("    Peso [N]   : %s" % _v(W_empty))
print("    Peso [kg]  : %s" % _v(W_empty / gravity))
print("    %% MTOW    : %s" % _v(W_empty / W0 * 100))
print()
print("  COMBUSTIVEL (W_fuel)")
print("    Peso [N]   : %s" % _v(W_fuel))
print("    Peso [kg]  : %s" % _v(W_fuel / gravity))
print("    %% MTOW    : %s" % _v(W_fuel / W0 * 100))
print()
print("  PAYLOAD (W_payload)")
print("    Peso [N]   : %s" % _v(inp['W_payload']))
print("    Peso [kg]  : %s" % _v(inp['W_payload'] / gravity))
print("    %% MTOW    : %s" % _v(inp['W_payload'] / W0 * 100))
print()
print("  TRIPULACAO (W_crew)")
print("    Peso [N]   : %s" % _v(inp['W_crew']))
print("    Peso [kg]  : %s" % _v(inp['W_crew'] / gravity))
print("    %% MTOW    : %s" % _v(inp['W_crew'] / W0 * 100))
print()
print("  MTOW (W0)")
print("    Peso [N]   : %s" % _v(W0))
print("    Peso [kg]  : %s" % _v(W0 / gravity))
print("    %% MTOW    : %s" % _v(100.0))
print()

# ============================================================
# 3. CONSUMO DE COMBUSTIVEL NAS ETAPAS DA MISSAO
# ============================================================
mf = airplane['fuel_weight']['Mf_hist']
ld = airplane['fuel_weight']['LD_hist']
tsfc = airplane['fuel_weight']['C_hist']
trapped = airplane['fuel_weight']['trapped_fuel_factor']

phases = ['start', 'taxi', 'takeoff', 'climb', 'cruise', 'descent', 'altcruise', 'loiter', 'landing']
phase_names = {
    'start': 'Partida dos motores',
    'taxi': 'Taxi',
    'takeoff': 'Decolagem',
    'climb': 'Subida',
    'cruise': 'Cruzeiro (8000 nmi)',
    'descent': 'Descida',
    'altcruise': 'Cruzeiro alternativo (200 nmi)',
    'loiter': 'Espera / Loiter (45 min)',
    'landing': 'Pouso',
}

print("=" * 70)
print("  3. CONSUMO DE COMBUSTIVEL POR ETAPA DA MISSAO")
print("=" * 70)
print()
print("  Perfil da missao:")
print("  partida -> taxi -> decolagem -> subida -> cruzeiro -> descida")
print("          -> cruzeiro alternativo -> loiter -> pouso")
print()

W_start = W0
Mf_total = 1.0

for phase in phases:
    frac = mf[phase]
    W_end = W_start * frac
    fuel_burned = (W_start - W_end) / gravity
    Mf_total *= frac
    print("  %s" % phase_names[phase])
    print("    Mf           : %s" % _v(frac))
    print("    W_inicio [kg]: %s" % _v(W_start / gravity))
    print("    Comb. [kg]   : %s" % _v(fuel_burned))
    print()
    W_start = W_end

fuel_mission = (1 - Mf_total) * W0 / gravity
fuel_trapped = (trapped - 1) * (1 - Mf_total) * W0 / gravity
print("  Fracao de massa total (Mf)          : %s" % _v(Mf_total))
print("  Combustivel da missao               : %s kg" % _v(fuel_mission))
print("  Combustivel preso (fator %s)        : %s kg" % (_v(trapped), _v(fuel_trapped)))
print("  Combustivel total (W_fuel)          : %s kg" % _v(W_fuel / gravity))
print()
print("  --- Eficiencia aerodinamica e TSFC por fase ---")
for phase in ['cruise', 'altcruise', 'loiter']:
    print("  %s" % phase_names[phase])
    print("    L/D        : %s" % _v(ld[phase]))
    print("    TSFC [1/s] : %s" % _v(tsfc[phase]))
    print()
print("  W_cruise (peso no inicio do cruzeiro): %s N (%s kg)" % (_v(W_cruise), _v(W_cruise / gravity)))
print()

# ============================================================
# 4. TRACAO
# ============================================================
from designTool.auxiliary import atmosphere
from designTool.propulsion import engineTSFC

altitude_cruise = inp['altitude_cruise']
Mach_cruise = inp['Mach_cruise']
S_w = inp['S_w']

atm_cr = atmosphere(altitude_cruise)
rho_cr = atm_cr['density']
a_cr = atm_cr['speed_of_sound']
V_cr = Mach_cruise * a_cr

CL_cr = 2.0 * W_cruise / (rho_cr * S_w * V_cr**2)
CD_cr = CL_cr / ld['cruise']
T_cruise = 0.5 * rho_cr * V_cr**2 * S_w * CD_cr

_, kT = engineTSFC(Mach_cruise, altitude_cruise, airplane)
T0_available = T0_guess
T_cruise_available = kT * T0_available

print("=" * 70)
print("  4. TRACAO")
print("=" * 70)
print()
print("  T0 (tracao inicial, SL estatico) : %s N (%s kN)" % (_v(T0_guess), _v(T0_guess / 1000)))
print("  T_cruise (tracao requerida)      : %s N (%s kN)" % (_v(T_cruise), _v(T_cruise / 1000)))
print("  kT (fator de correcao)           : %s" % _v(kT))
print("  T_cruise_disponivel (kT * T0)    : %s N (%s kN)" % (_v(T_cruise_available), _v(T_cruise_available / 1000)))
print("  T_cruise / T0                    : %s" % _v(T_cruise / T0_guess))
print("  Margem de empuxo em cruzeiro     : %s %%" % _v((T_cruise_available / T_cruise - 1) * 100))
print()
