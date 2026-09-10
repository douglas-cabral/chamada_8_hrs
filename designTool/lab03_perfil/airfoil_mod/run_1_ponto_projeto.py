'''
PRJ-23 Lab 03 - Ponto de projeto (meio do cruzeiro) da NJ-0502 otimizada.
Uso: python run_1_ponto_projeto.py
'''

import copy
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_LAB)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from designTool.analyze import analyze
from designTool.auxiliary import atmosphere
from designTool.constants import ft2m, gravity
from designTool.standard_airplane import standard_airplane
from designTool.weight import fuel_weight

RESULTS_DIR = os.path.join(_LAB, 'resultados_airfoil_mod')
TEX_DIR = os.path.join(_LAB, 'tex_airfoil_mod')
N_REFINE = 4


def fmt_br(x, nd):
    return (('%.' + str(nd) + 'f') % x).replace('.', '{,}')


def fmt_kgf(x):
    s = '%.1f' % x
    inteiro, frac = s.split('.')
    grupos = []
    while inteiro:
        grupos.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    return r'\,'.join(reversed(grupos)) + '{,}' + frac


def run():
    airplane = {'inputs': copy.deepcopy(standard_airplane('my_airplane')['inputs'])}
    analyze(airplane, print_log=False, plot=False)
    for _ in range(N_REFINE):
        airplane['inputs']['W0_guess'] = airplane['thrust_matching']['W0']
        analyze(airplane, print_log=False, plot=False)

    W0 = airplane['thrust_matching']['W0']
    _, W_ini = fuel_weight(
        W0, airplane,
        range_cruise=airplane['inputs']['range_cruise'],
        update_Mf_hist=True)
    Mf_c = airplane['fuel_weight']['Mf_hist']['cruise']
    W_fim = W_ini*Mf_c
    W = 0.5*(W_ini + W_fim)

    h = airplane['inputs']['altitude_cruise']
    M = airplane['inputs']['Mach_cruise']
    S = airplane['inputs']['S_w']
    atm = atmosphere(h)
    rho = atm['density']
    a = atm['speed_of_sound']
    mi = atm['dyn_viscosity']
    v = M*a
    c_ref = airplane['geometry']['cm_w']
    clref = 2.0*W/(rho*S*v**2)
    Re = rho*v*c_ref/mi
    tcr = airplane['inputs']['tcr_w']
    tct = airplane['inputs']['tct_w']
    tc_ref = 0.5*(tcr + tct)

    return {
        'h_ft': h/ft2m,
        'M': M,
        'W_kgf': W/gravity,
        'S_ref': S,
        'rho': rho,
        'a': a,
        'c_ref': c_ref,
        'clref': clref,
        'Re_ref': Re,
        'tc_ref': tc_ref,
        'W0_kgf': W0/gravity,
        'W_ini_kgf': W_ini/gravity,
        'W_fim_kgf': W_fim/gravity,
        'Mf_cruise': Mf_c,
        'v': v,
        'mi': mi,
        'tcr': tcr,
        'tct': tct,
    }


def write_csv(rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'ponto_projeto.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['parametro', 'valor'])
        for k, v in rows.items():
            w.writerow([k, v])
    print('  gravado:', path)


def fmt_int_milhar(x):
    inteiro = '%d' % round(x)
    grupos = []
    while inteiro:
        grupos.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    return r'\,'.join(reversed(grupos))


def write_tex(rows):
    os.makedirs(TEX_DIR, exist_ok=True)
    mant, exp = ('%.3e' % rows['Re_ref']).split('e')
    re_tex = r'%s \times 10^{%d}' % (mant.replace('.', '{,}'), int(exp))
    lines = [
        r'\begin{tabular}{lll}',
        r'\toprule',
        r'Parâmetro & Valor & Descrição \\',
        r'\midrule',
        r'$h$ & $%s$ & Altitude do ponto de projeto [ft] \\'
        % fmt_int_milhar(rows['h_ft']),
        r'$M_{\infty}$ & $%s$ & Mach do ponto de projeto \\'
        % fmt_br(rows['M'], 2),
        r'$W$ & $%s$ & Peso da aeronave no ponto de projeto [kgf] \\'
        % fmt_kgf(rows['W_kgf']),
        r'$S_{\mathrm{ref}}$ & $%s$ & Área de referência da aeronave [m$^2$] \\'
        % fmt_br(rows['S_ref'], 2),
        r'$\rho_{\infty}$ & $%s$ & Densidade do ar no ponto de projeto [kg/m$^3$] \\'
        % fmt_br(rows['rho'], 4),
        r'$a_{\infty}$ & $%s$ & Velocidade do som no ponto de projeto [m/s] \\'
        % fmt_br(rows['a'], 2),
        r'$c_{\mathrm{ref}}$ & $%s$ & Corda da seção de referência [m] \\'
        % fmt_br(rows['c_ref'], 3),
        r'$c_{l\mathrm{ref}}$ & $%s$ & Coeficiente de sustentação da seção de referência \\'
        % fmt_br(rows['clref'], 4),
        r'$Re_{\mathrm{ref}}$ & $%s$ & Reynolds para a seção de referência \\'
        % re_tex,
        r'$(t/c)_{\mathrm{ref}}$ & $%s$ & Espessura relativa base da seção de referência \\'
        % fmt_br(rows['tc_ref'], 2),
        r'\bottomrule',
        r'\end{tabular}',
    ]
    path = os.path.join(TEX_DIR, 'tab_ponto_projeto.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('  gravado:', path)


if __name__ == '__main__':
    rows = run()
    write_csv(rows)
    write_tex(rows)
    print('  W0=%.1f  W_ini=%.1f  W_fim=%.1f  W_mid=%.1f kgf'
          % (rows['W0_kgf'], rows['W_ini_kgf'], rows['W_fim_kgf'],
             rows['W_kgf']))
    print('  CL=%.4f  Re=%.3e  tc=%.2f'
          % (rows['clref'], rows['Re_ref'], rows['tc_ref']))
