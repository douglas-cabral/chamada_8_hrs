'''
PRJ-23 Lab 02 - Problema 3. Tabelas LaTeX a partir dos CSV.
Uso: python gera_tex.py
'''

import os

import numpy as np
import pandas as pd

# =========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_LAB, 'resultados_otimizacao_NJ0502')
TEX_DIR = os.path.join(_LAB, 'tex_otimizacao_NJ0502')


def esc(s):
    return str(s).replace('_', r'\_')


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    return (('%.' + str(nd) + 'f') % v).replace('-', r'$-$')


def write(name, content):
    path = os.path.join(TEX_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('  gravado: %s' % path)


# =========================================


def tab_variaveis():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_variaveis.csv'),
                     index_col=0)
    lines = [r'\begin{tabular}{lrrrrr}',
             r'\toprule',
             r"Variável & Inicial & Ótimo & Lim.\ inf.\ & "
             r'Lim.\ sup.\ & $\Delta$ [\%] \\',
             r'\midrule']
    for _, row in df.iterrows():
        lines.append(
            '%s [%s] & %s & %s & %s & %s & %s \\\\' % (
                row['label'], row['unidade'],
                fmt(row['inicial'], 4), fmt(row['otimo'], 4),
                fmt(row['lim_inf'], 4), fmt(row['lim_sup'], 4),
                fmt(row['variacao_pct'], 2)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_restricoes():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_restricoes.csv'),
                     index_col=0)
    lines = [r'\begin{tabular}{llrrr}',
             r'\toprule',
             r'Restrição & $g(x)$ & $g_{\mathrm{ini}}$ & '
             r'$g_{\mathrm{ot}}$ & Ativa? \\',
             r'\midrule']
    for _, row in df.iterrows():
        ativa = r'\textbf{sim}' if bool(row['ativa']) else r'não'
        lines.append('%s & %s & %s & %s & %s \\\\' % (
            row['descricao'], row['expressao'],
            fmt(row['g_inicial'], 4), fmt(row['g_otimo'], 4), ativa))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_grandezas():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_grandezas.csv'),
                     index_col=0)
    keep = [
        'W0 [kgf]', 'W_empty [kgf]', 'W_fuel [kgf]',
        'T0 [kgf]', 'T0req [kgf]',
        'S_w [m2]', 'b_w [m]', 'S_h [m2]', 'S_v [m2]',
        'deltaS_wlan [m2]', 'SM_fwd [-]', 'SM_aft [-]', 'CLv [-]',
        'tank_excess [-]',
        'frac_nlg_fwd [-]', 'frac_nlg_aft [-]',
        'alpha_tipback [deg]', 'alpha_tail [deg]', 'phi_overturn [deg]',
        'h_tail [m]', 'wheel_span [m]', 'xi_mlg [-]',
        'L_f [m]', 'x_te_v [m]', 'Lb_v [-]',
        'x_te_h [m]', 'Lc_h [-]',
    ]
    labels = {
        'W0 [kgf]': r'$W_0$ [kgf]',
        'W_empty [kgf]': r'$W_e$ [kgf]',
        'W_fuel [kgf]': r'$W_f$ [kgf]',
        'T0 [kgf]': r'$T_0$ [kgf]',
        'T0req [kgf]': r'$T_{0,\mathrm{req}}$ [kgf]',
        'S_w [m2]': r'$S_w$ [m$^2$]',
        'b_w [m]': r'$b_w$ [m]',
        'S_h [m2]': r'$S_h$ [m$^2$]',
        'S_v [m2]': r'$S_v$ [m$^2$]',
        'deltaS_wlan [m2]': r'$\Delta S_{wlan}$ [m$^2$]',
        'SM_fwd [-]': r'$SM_{fwd}$',
        'SM_aft [-]': r'$SM_{aft}$',
        'CLv [-]': r'$C_{Lv}$',
        'tank_excess [-]': r'$tank\_excess$',
        'frac_nlg_fwd [-]': r'$f_{nlg,fwd}$',
        'frac_nlg_aft [-]': r'$f_{nlg,aft}$',
        'alpha_tipback [deg]': r'$\alpha_{tip}$ [deg]',
        'alpha_tail [deg]': r'$\alpha_{tail}$ [deg]',
        'phi_overturn [deg]': r'$\phi_{ovt}$ [deg]',
        'h_tail [m]': r'$h_{tail}$ [m]',
        'wheel_span [m]': r'$b_{mlg}$ [m]',
        'xi_mlg [-]': r'$\xi_{mlg}$',
        'L_f [m]': r'$L_f$ [m]',
        'x_te_v [m]': r'$x_{\mathrm{TE},v}$ [m]',
        'Lb_v [-]': r'$L_{b,v}$',
        'x_te_h [m]': r'$x_{\mathrm{TE},h}$ [m]',
        'Lc_h [-]': r'$L_{c,h}$',
    }
    lines = [r'\begin{tabular}{lrrr}',
             r'\toprule',
             r"Grandeza & Inicial & Ótimo & $\Delta$ [\%] \\",
             r'\midrule']
    for key in keep:
        row = df.loc[key]
        lines.append('%s & %s & %s & %s \\\\' % (
            labels[key],
            fmt(row['inicial'], 3),
            fmt(row['otimo'], 3),
            fmt(row['variacao_pct'], 2)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


if __name__ == '__main__':

    os.makedirs(TEX_DIR, exist_ok=True)

    write('tab_variaveis.tex', tab_variaveis())
    write('tab_restricoes.tex', tab_restricoes())
    write('tab_grandezas.tex', tab_grandezas())
