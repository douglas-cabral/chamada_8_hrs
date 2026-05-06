# Script de estimativa de peso W0 para a aeronave de projeto.

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.weight import weight
from designTool.constants import gravity, nm2m, ft2m
import numpy as np

# Load airplane
airplane = standard_airplane('my_airplane')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration
W0_guess = airplane['inputs']['W0_guess']
T0_guess = 850000.0

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
print("  Numero de motores      : %d" % inp['n_engines'])
print()
print("  --- Missao de Projeto ---")
print("  Alcance de cruzeiro    : %.0f nmi" % (inp['range_cruise'] / nm2m))
print("  Mach de cruzeiro       : %.2f" % inp['Mach_cruise'])
print("  Altitude de cruzeiro   : %.0f ft (%.0f m)" % (inp['altitude_cruise'] / ft2m, inp['altitude_cruise']))
print("  Mach maximo operacional: %.2f" % inp['Mach_maxcruise'])
print("  Altitude max cruzeiro  : %.0f ft (%.0f m)" % (inp['altitude_maxcruise'] / ft2m, inp['altitude_maxcruise']))
print()
print("  --- Decolagem ---")
print("  Comprimento de pista   : %.0f m" % inp['distance_takeoff'])
print("  Altitude               : %.0f m (nivel do mar)" % inp['altitude_takeoff'])
print("  Delta ISA              : %.1f C (condicoes ISA)" % inp['deltaISA_takeoff'])
print()
print("  --- Reservas ---")
print("  Tempo de espera (loiter): %.0f min" % (inp['time_loiter'] / 60))
print("  Altitude de loiter      : %.0f ft (%.0f m)" % (inp['altitude_loiter'] / ft2m, inp['altitude_loiter']))
print("  Cruzeiro alternativo    : %.0f nmi" % (inp['range_altcruise'] / nm2m))
print("  Mach alternativo        : %.2f" % inp['Mach_altcruise'])
print("  Altitude alternativa    : %.0f m (%.0f ft)" % (inp['altitude_altcruise'], inp['altitude_altcruise'] / ft2m))
print()
print("  --- Pesos de Entrada ---")
print("  W_payload (320x100 kg) : %.0f N (%.0f kg)" % (inp['W_payload'], inp['W_payload'] / gravity))
print("  W_crew                 : %.0f N (%.0f kg)" % (inp['W_crew'], inp['W_crew'] / gravity))
print()

# ============================================================
# 2. BREAKDOWN DE PESO
# ============================================================
ew = airplane['empty_weight']

print("=" * 70)
print("  2. BREAKDOWN DE PESO DA AERONAVE")
print("=" * 70)
print()
print("  %-28s  %12s  %12s  %8s" % ("Componente", "Peso [N]", "Peso [kg]", "% MTOW"))
print("  " + "-" * 64)

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
    print("  %-28s  %12.1f  %12.1f  %7.2f%%" % (name, val, val / gravity, val / W0 * 100))

print("  " + "-" * 64)
print("  %-28s  %12.1f  %12.1f  %7.2f%%" % ("PESO VAZIO (W_empty)", W_empty, W_empty / gravity, W_empty / W0 * 100))
print("  %-28s  %12.1f  %12.1f  %7.2f%%" % ("COMBUSTIVEL (W_fuel)", W_fuel, W_fuel / gravity, W_fuel / W0 * 100))
print("  %-28s  %12.1f  %12.1f  %7.2f%%" % ("PAYLOAD (W_payload)", inp['W_payload'], inp['W_payload'] / gravity, inp['W_payload'] / W0 * 100))
print("  %-28s  %12.1f  %12.1f  %7.2f%%" % ("TRIPULACAO (W_crew)", inp['W_crew'], inp['W_crew'] / gravity, inp['W_crew'] / W0 * 100))
print("  " + "-" * 64)
print("  %-28s  %12.1f  %12.1f  %7.2f%%" % ("MTOW (W0)", W0, W0 / gravity, 100.0))
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
print("  %-32s  %10s  %12s  %12s" % ("Etapa", "Mf", "W_inicio [kg]", "Comb. [kg]"))
print("  " + "-" * 70)

W_start = W0
Mf_total = 1.0

for phase in phases:
    frac = mf[phase]
    W_end = W_start * frac
    fuel_burned = (W_start - W_end) / gravity
    Mf_total *= frac
    print("  %-32s  %10.6f  %12.1f  %12.1f" % (
        phase_names[phase], frac, W_start / gravity, fuel_burned))
    W_start = W_end

print("  " + "-" * 70)
fuel_mission = (1 - Mf_total) * W0 / gravity
fuel_trapped = (trapped - 1) * (1 - Mf_total) * W0 / gravity
print()
print("  Fracao de massa total (Mf)          : %.6f" % Mf_total)
print("  Combustivel da missao               : %.1f kg" % fuel_mission)
print("  Combustivel preso (fator %.2f)      : %.1f kg" % (trapped, fuel_trapped))
print("  Combustivel total (W_fuel)          : %.1f kg" % (W_fuel / gravity))
print()
print("  --- Eficiencia aerodinamica e TSFC por fase ---")
print("  %-32s  %10s  %16s" % ("Etapa", "L/D", "TSFC [1/s]"))
print("  " + "-" * 60)
for phase in ['cruise', 'altcruise', 'loiter']:
    print("  %-32s  %10.4f  %16.10f" % (phase_names[phase], ld[phase], tsfc[phase]))
print()
print("  W_cruise (peso no inicio do cruzeiro): %.1f N (%.1f kg)" % (W_cruise, W_cruise / gravity))
print()
