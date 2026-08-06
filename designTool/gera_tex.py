'''
INSTITUTO TECNOLOGICO DE AERONAUTICA
PRJ-23 - Homework 01 - DOE analysis - Grupo NJ-0502

Gera os fragmentos LaTeX consumidos por relatorio.tex a partir dos CSV
produzidos pelos tres scripts de analise. Assim, qualquer reexecucao dos
scripts se propaga automaticamente para o relatorio.

Uso:  python gera_tex.py     (apos rodar os tres scripts run_*.py)
'''

# IMPORTS
import os

import numpy as np
import pandas as pd

from doe_common import get_baseline, run_baseline
from doe_common import rad2deg
from designTool.constants import ft2m, nm2m, gravity, lb2N

RESULTS_DIR = 'resultados'
TEX_DIR = 'tex'

# =========================================


def esc(s):
    '''Escapa underscores em nomes de variaveis do Python para LaTeX.'''
    return s.replace('_', r'\_')


def tab_ranking(n=20):
    '''Tabela do ranking de triagem (Atividade 2.4).'''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'triagem_ranking.csv'),
                     index_col=0)
    df = df.head(n)

    lines = []
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        lines.append('%d & %s & %.2f & %s & %d & %.2f \\\\'
                     % (i, idx, row['S_max'], row['saida_dominante'],
                        int(row['n_saidas_relevantes']), row['S_medio']))

    return '\n'.join([
        r'\begin{tabular}{rlrlrr}',
        r'\toprule',
        r'\# & Entrada & $\max_i|S_{ij}|$ & Sa\'ida dominante & '
        r'$n_{|S|\geq 0{,}1}$ & $\overline{|S_{ij}|}$ \\',
        r'\midrule',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{tabular}',
    ])


def tab_corr():
    '''Matriz de Pearson do DOE de 400 amostras (Atividade 2.3).'''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'correlacoes_400.csv'),
                     index_col=0)

    short = {c: c.split(' [')[0] for c in df.columns}
    df = df.rename(index=short, columns=short)

    lines = []
    for idx, row in df.iterrows():
        cells = ['%.3f' % v for v in row.values]
        cells = [c.replace('-', '$-$') for c in cells]
        lines.append('%s & %s \\\\' % (idx, ' & '.join(cells)))

    return '\n'.join([
        r'\begin{tabular}{l|' + 'r' * df.shape[1] + '}',
        r'\toprule',
        ' & ' + ' & '.join(df.columns) + r' \\',
        r'\midrule',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{tabular}',
    ])


def tab_passo():
    '''Estudo de independencia do passo (verificacao de consistencia).'''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'sens_passo_my_airplane.csv'),
                     index_col=0)

    lines = []
    for idx, row in df.iterrows():
        cells = ['%.4f' % v for v in row.values]
        cells = [c.replace('-', '$-$') for c in cells]
        lines.append('%s & %s \\\\' % (idx, ' & '.join(cells)))

    # '%' e caractere de comentario no LaTeX e aparece nos nomes das colunas
    cols = [c.replace('%', r'\%') for c in df.columns]

    return '\n'.join([
        r'\begin{tabular}{l' + 'r' * df.shape[1] + '}',
        r'\toprule',
        r'Entrada & ' + ' & '.join(cols) + r' \\',
        r'\midrule',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{tabular}',
    ])


def tab_passos():
    '''
    Denominador da tabela do enunciado: passo relativo dX/X* aplicado a cada
    entrada, nas duas aeronaves. Torna verificavel que as perturbacoes
    absolutas (+2 deg em dihedral_w, +0,02 em Mach) foram normalizadas pelo
    valor de referencia de CADA aeronave, e nao usadas cruas.
    '''
    fk = pd.read_csv(os.path.join(RESULTS_DIR, 'passos_fokker100.csv'),
                     index_col=0)
    my = pd.read_csv(os.path.join(RESULTS_DIR, 'passos_my_airplane.csv'),
                     index_col=0)

    # Angulos ficam mais legiveis em graus
    ang = {r'$\Lambda_w$': rad2deg, r'$\delta_w$': rad2deg}
    unit = {r'$\Lambda_w$': 'deg', r'$\delta_w$': 'deg',
            r'$R$': 'nm', r'$S_w$': 'm$^2$', r'$x_{r,w}$': 'm'}
    conv = {r'$R$': 1.0 / nm2m}

    lines = []
    for idx in fk.index:
        f = rad2deg if idx in ang else conv.get(idx, 1.0)
        lines.append(
            '%s & %s & %s & %.3f & %.5f & %.3f & %.5f \\\\'
            % (idx, unit.get(idx, '--'),
               # '%' e caractere de comentario no LaTeX
               str(fk.loc[idx, 'perturbacao']).replace('%', r'\%')
                                              .replace(' deg', r'$^{\circ}$'),
               fk.loc[idx, 'X_ref'] * f, fk.loc[idx, 'dX/X*'],
               my.loc[idx, 'X_ref'] * f, my.loc[idx, 'dX/X*']))

    return '\n'.join([
        r'\begin{tabular}{llcrrrr}',
        r'\toprule',
        r' & & & \multicolumn{2}{c}{Fokker 100} & '
        r'\multicolumn{2}{c}{NJ-0502} \\',
        r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
        r'Entrada & Un. & $\Delta X$ & $X^*$ & $\Delta X/X^*$ & '
        r'$X^*$ & $\Delta X/X^*$ \\',
        r'\midrule',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{tabular}',
    ])


def tab_baseline():
    '''Saidas convergidas das duas aeronaves de referencia.'''
    rows = [
        ('$W_0$ [kgf]',                 'W0',               '%.0f'),
        ('$W_f$ [kgf]',                 'W_f',              '%.0f'),
        ('$W_e$ [kgf]',                 'W_e',              '%.0f'),
        ('$T_0$ [kgf]',                 'T0',               '%.0f'),
        ('$\\Delta S_{wlan}$ [m$^2$]',  'deltaS_wlan',      '%.1f'),
        ('$SM_{fwd}$',                  'SM_fwd',           '%.4f'),
        ('$SM_{aft}$',                  'SM_aft',           '%.4f'),
        ('$C_{Lv}$',                    'CLv',              '%.4f'),
        ('$V_{tank}$ [L]',              'V_maxfuel',        '%.0f'),
        ('sobra de tanque [\\%]',       'tank_excess',      '%.1f'),
        ('$f_{nlg,fwd}$ [\\%]',         'frac_nlg_fwd',     '%.1f'),
        ('$f_{nlg,aft}$ [\\%]',         'frac_nlg_aft',     '%.1f'),
        ('$\\alpha_{tipback}$ [$^\\circ$]',    'alpha_tipback',    '%.1f'),
        ('$\\alpha_{tailstrike}$ [$^\\circ$]', 'alpha_tailstrike', '%.1f'),
        ('$\\phi_{overturn}$ [$^\\circ$]',     'phi_overturn',     '%.1f'),
    ]
    pct = {'tank_excess', 'frac_nlg_fwd', 'frac_nlg_aft'}

    fk = run_baseline('fokker100')
    my = run_baseline('my_airplane')

    lines = []
    for label, key, fmt in rows:
        a = fk[key] * (100.0 if key in pct else 1.0)
        b = my[key] * (100.0 if key in pct else 1.0)
        lines.append('%s & %s & %s \\\\'
                     % (label,
                        (fmt % a).replace('-', '$-$'),
                        (fmt % b).replace('-', '$-$')))

    return '\n'.join([
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r'Sa\'ida & Fokker 100 & Aeronave NJ-0502 \\',
        r'\midrule',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{tabular}',
    ])


def tab_dicionario():
    '''
    Dicionario de entrada da aeronave do grupo (deliverable do enunciado).
    Angulos convertidos para graus e comprimentos de missao para unidades
    de engenharia, para leitura direta.
    '''
    inp = get_baseline('my_airplane')['inputs']

    grupos = [
        ('Asa', [
            ('S_w',        'm$^2$',   inp['S_w'],                    '%.2f'),
            ('AR_w',       '--',      inp['AR_w'],                   '%.2f'),
            ('taper_w',    '--',      inp['taper_w'],                '%.2f'),
            ('sweep_w',    'deg',     inp['sweep_w'] * rad2deg,      '%.2f'),
            ('dihedral_w', 'deg',     inp['dihedral_w'] * rad2deg,   '%.2f'),
            ('xr_w',       'm',       inp['xr_w'],                   '%.2f'),
            ('zr_w',       'm',       inp['zr_w'],                   '%.2f'),
            ('tcr_w',      '--',      inp['tcr_w'],                  '%.3f'),
            ('tct_w',      '--',      inp['tct_w'],                  '%.3f'),
            ('clmax_w',    '--',      inp['clmax_w'],                '%.2f'),
            ('k_korn',     '--',      inp['k_korn'],                 '%.2f'),
            ('winglet',    '--',      float(inp['winglet']),         '%.0f'),
        ]),
        ('Empenagens', [
            ('Cht',        '--',      inp['Cht'],                    '%.3f'),
            ('Lc_h',       '--',      inp['Lc_h'],                   '%.2f'),
            ('AR_h',       '--',      inp['AR_h'],                   '%.2f'),
            ('taper_h',    '--',      inp['taper_h'],                '%.2f'),
            ('sweep_h',    'deg',     inp['sweep_h'] * rad2deg,      '%.2f'),
            ('dihedral_h', 'deg',     inp['dihedral_h'] * rad2deg,   '%.2f'),
            ('zr_h',       'm',       inp['zr_h'],                   '%.2f'),
            ('tcr_h',      '--',      inp['tcr_h'],                  '%.3f'),
            ('tct_h',      '--',      inp['tct_h'],                  '%.3f'),
            ('eta_h',      '--',      inp['eta_h'],                  '%.2f'),
            ('Cvt',        '--',      inp['Cvt'],                    '%.3f'),
            ('Lb_v',       '--',      inp['Lb_v'],                   '%.2f'),
            ('AR_v',       '--',      inp['AR_v'],                   '%.2f'),
            ('taper_v',    '--',      inp['taper_v'],                '%.2f'),
            ('sweep_v',    'deg',     inp['sweep_v'] * rad2deg,      '%.2f'),
            ('zr_v',       'm',       inp['zr_v'],                   '%.2f'),
            ('tcr_v',      '--',      inp['tcr_v'],                  '%.3f'),
            ('tct_v',      '--',      inp['tct_v'],                  '%.3f'),
        ]),
        ('Fuselagem, nacele e motor', [
            ('L_f',        'm',       inp['L_f'],                    '%.2f'),
            ('D_f',        'm',       inp['D_f'],                    '%.2f'),
            ('x_n',        'm',       inp['x_n'],                    '%.2f'),
            ('y_n',        'm',       inp['y_n'],                    '%.2f'),
            ('z_n',        'm',       inp['z_n'],                    '%.2f'),
            ('L_n',        'm',       inp['L_n'],                    '%.2f'),
            ('D_n',        'm',       inp['D_n'],                    '%.2f'),
            ('n_engines',  '--',      inp['n_engines'],              '%.0f'),
            ('n_engines_under_wing', '--',
                                      inp['n_engines_under_wing'],   '%.0f'),
            ('engine.BPR', '--',      inp['engine']['BPR'],          '%.2f'),
            ('engine.weight', 'kgf',
                            inp['engine']['weight'] / gravity,       '%.0f'),
            ('engine.Tmax', 'lbf',
                            inp['engine']['Tmax'] / lb2N,            '%.0f'),
            ('engine.C_ref', '1/h',
                            inp['engine']['C_ref'] * 3600,           '%.3f'),
            ('engine.Mach_ref', '--',
                            inp['engine']['Mach_ref'],               '%.2f'),
            ('engine.altitude_ref', 'ft',
                            inp['engine']['altitude_ref'] / ft2m,    '%.0f'),
        ]),
        ('Trem de pouso', [
            ('x_nlg',         'm', inp['x_nlg'],                     '%.2f'),
            ('x_mlg',         'm', inp['x_mlg'],                     '%.2f'),
            ('y_mlg',         'm', inp['y_mlg'],                     '%.2f'),
            ('z_lg',          'm', inp['z_lg'],                      '%.2f'),
            ('x_tailstrike',  'm', inp['x_tailstrike'],              '%.2f'),
            ('z_tailstrike',  'm', inp['z_tailstrike'],              '%.2f'),
        ]),
        ('Tanque e hipersustentadores', [
            ('c_tank_c_w',       '--', inp['c_tank_c_w'],            '%.2f'),
            ('x_tank_c_w',       '--', inp['x_tank_c_w'],            '%.2f'),
            ('b_tank_b_w_start', '--', inp['b_tank_b_w_start'],      '%.2f'),
            ('b_tank_b_w_end',   '--', inp['b_tank_b_w_end'],        '%.2f'),
            ('flap_type',        '--', inp['flap_type'],             '%s'),
            ('c_flap_c_wing',    '--', inp['c_flap_c_wing'],         '%.2f'),
            ('b_flap_b_wing',    '--', inp['b_flap_b_wing'],         '%.2f'),
            ('slat_type',        '--', inp['slat_type'],             '%s'),
            ('c_slat_c_wing',    '--', inp['c_slat_c_wing'],         '%.2f'),
            ('b_slat_b_wing',    '--', inp['b_slat_b_wing'],         '%.2f'),
            ('c_ail_c_wing',     '--', inp['c_ail_c_wing'],          '%.2f'),
            ('b_ail_b_wing',     '--', inp['b_ail_b_wing'],          '%.2f'),
            ('k_exc_drag',       '--', inp['k_exc_drag'],            '%.2f'),
            ('h_ground',         'm',  inp['h_ground'],              '%.2f'),
        ]),
        ('Missao e requisitos', [
            ('altitude_cruise',   'ft', inp['altitude_cruise'] / ft2m, '%.0f'),
            ('Mach_cruise',       '--', inp['Mach_cruise'],            '%.2f'),
            ('range_cruise',      'nm', inp['range_cruise'] / nm2m,    '%.0f'),
            ('altitude_maxcruise', 'ft',
                                  inp['altitude_maxcruise'] / ft2m,    '%.0f'),
            ('Mach_maxcruise',    '--', inp['Mach_maxcruise'],         '%.2f'),
            ('altitude_altcruise', 'ft',
                                  inp['altitude_altcruise'] / ft2m,    '%.0f'),
            ('Mach_altcruise',    '--', inp['Mach_altcruise'],         '%.2f'),
            ('range_altcruise',   'nm', inp['range_altcruise'] / nm2m, '%.0f'),
            ('time_loiter',       'min', inp['time_loiter'] / 60,      '%.0f'),
            ('altitude_loiter',   'ft', inp['altitude_loiter'] / ft2m, '%.0f'),
            ('distance_takeoff',  'm',  inp['distance_takeoff'],       '%.0f'),
            ('altitude_takeoff',  'm',  inp['altitude_takeoff'],       '%.0f'),
            ('deltaISA_takeoff',  'C',  inp['deltaISA_takeoff'],       '%.0f'),
            ('distance_landing',  'm',  inp['distance_landing'],       '%.0f'),
            ('altitude_landing',  'm',  inp['altitude_landing'],       '%.0f'),
            ('deltaISA_landing',  'C',  inp['deltaISA_landing'],       '%.0f'),
            ('MLW_frac',          '--', inp['MLW_frac'],               '%.3f'),
        ]),
        ('Carga paga, tripulacao e operacao', [
            ('W_payload',   'kgf', inp['W_payload'] / gravity,      '%.0f'),
            ('xcg_payload', 'm',   inp['xcg_payload'],              '%.2f'),
            ('W_crew',      'kgf', inp['W_crew'] / gravity,         '%.0f'),
            ('xcg_crew',    'm',   inp['xcg_crew'],                 '%.2f'),
            ('block_range', 'nm',  inp['block_range'] / nm2m,       '%.0f'),
            ('block_time',  'h',   inp['block_time'] / 3600,        '%.2f'),
            ('n_captains',  '--',  inp['n_captains'],               '%.0f'),
            ('n_copilots',  '--',  inp['n_copilots'],               '%.0f'),
            ('rho_fuel',    'kg/m$^3$', inp['rho_fuel'],            '%.0f'),
            ('W0_guess',    'N',   inp['W0_guess'],                 '%.0f'),
            ('type',        '--',  inp['type'],                     '%s'),
        ]),
    ]

    lines = []
    for titulo, itens in grupos:
        lines.append(r'\midrule')
        lines.append(r'\multicolumn{3}{l}{\textbf{%s}} \\' % titulo)
        lines.append(r'\midrule')
        for key, unit, value, fmt in itens:
            if fmt == '%s':
                txt = esc(str(value))
            else:
                txt = fmt % value
            lines.append(r'\texttt{%s} & %s & %s \\'
                         % (esc(key), unit, txt))

    head = (r'\textbf{Chave (\texttt{designTool})} & \textbf{Unidade} & '
            r'\textbf{Valor} \\')

    return '\n'.join([
        r'\begin{longtable}{lll}',
        "\\caption{Dicion\\'ario de entrada da aeronave do grupo NJ-0502 "
        "(\\texttt{standard\\_airplane(\\textquotesingle my\\_airplane"
        "\\textquotesingle)}). \\^Angulos convertidos para graus e grandezas "
        "de miss\\~ao para unidades de engenharia.}"
        "\\label{tab:dic} \\\\",
        r'\toprule',
        head,
        r'\endfirsthead',
        r'\multicolumn{3}{l}{\small\emph{(continua\c{c}\~ao da '
        r'Tabela \ref{tab:dic})}} \\',
        r'\toprule',
        head,
        r'\endhead',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{longtable}',
    ])


def tab_triagem_completa():
    '''Matriz S_ij completa para o apendice.'''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'triagem_completa.csv'),
                     index_col=0)

    lines = []
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            if not np.isfinite(v):
                cells.append('--')
            elif abs(v) < 5e-4:
                cells.append('0')
            elif abs(v) >= 100:
                cells.append(('%.0f' % v).replace('-', '$-$'))
            else:
                cells.append(('%.2f' % v).replace('-', '$-$'))
        lines.append('%s & %s \\\\' % (idx, ' & '.join(cells)))

    return '\n'.join([
        r'\begin{longtable}{l|' + 'r' * df.shape[1] + '}',
        r'\toprule',
        'Entrada & ' + ' & '.join(df.columns) + r' \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        'Entrada & ' + ' & '.join(df.columns) + r' \\',
        r'\midrule',
        r'\endhead',
        '\n'.join(lines),
        r'\bottomrule',
        r'\end{longtable}',
    ])


# =========================================

if __name__ == '__main__':

    os.makedirs(TEX_DIR, exist_ok=True)

    fragments = {
        'tab_ranking.tex':          tab_ranking(),
        'tab_passos.tex':           tab_passos(),
        'tab_corr.tex':             tab_corr(),
        'tab_passo.tex':            tab_passo(),
        'tab_baseline.tex':         tab_baseline(),
        'tab_dicionario.tex':       tab_dicionario(),
        'tab_triagem_completa.tex': tab_triagem_completa(),
    }

    for name, content in fragments.items():
        with open(os.path.join(TEX_DIR, name), 'w') as f:
            f.write(content)
        print('  gravado: %s/%s' % (TEX_DIR, name))
