'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 2 - Grupo NJ-0502

Gera os fragmentos LaTeX consumidos por otimizacao_aeronave_padrao.tex a
partir dos CSV produzidos pelos scripts de otimização.

Uso:  python gera_tex.py     (após rodar run_2_1 e run_2_2)
'''

# IMPORTS
import os

import numpy as np
import pandas as pd

# =========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, 'resultados_otimizacao_aeronave_padrao')
TEX_DIR = os.path.join(_HERE, 'tex_otimizacao_aeronave_padrao')


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


def tab_resultados():
    '''
    Tabela 1 do roteiro: AR_w, S_w, MTOW e b_w na partida e no ótimo.
    '''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_tabela1.csv'),
                     index_col=0)
    lines = [r'\begin{tabular}{lrrrr}',
             r'\toprule',
             r'& $AR_w$ & $S_w$ [m$^2$] & MTOW [kgf] & $b_w$ [m] \\',
             r'\midrule']
    nomes = {'Partida': 'Ponto de partida', 'Ótimo': 'Ponto otimizado'}
    for idx, row in df.iterrows():
        lines.append('%s & %s & %s & %s & %s \\\\' % (
            nomes.get(idx, esc(idx)),
            fmt(row['AR_w'], 4), fmt(row['S_w'], 3),
            fmt(row['MTOW_kgf'], 1), fmt(row['b_w'], 3)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_corrida():
    '''
    Contadores da corrida do SLSQP.
    '''
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_corrida.csv'))
    r = df.iloc[0]
    itens = [
        (r'Iterações do SLSQP (\code{nit})', '%d' % r['n_iter']),
        (r'Avaliações da função objetivo (\code{nfev})', '%d' % r['nfev_scipy']),
        (r'Avaliações do gradiente (\code{njev})', '%d' % r['njev_scipy']),
        (r'Avaliações da restrição', '%d' % r['n_confun']),
        (r'Chamadas distintas ao \emph{Design Tools}', '%d' % r['n_designTool']),
        (r'Tempo de parede [s]', fmt(r['tempo_s'], 3)),
        (r'$W_0$ inicial [kgf]', fmt(r['W0_ini_kgf'], 2)),
        (r'$W_0$ ótimo [kgf]', fmt(r['W0_kgf'], 2)),
        (r'Melhoria relativa [\%]', fmt(r['ganho_pct'], 3)),
        (r'$b_w$ no ótimo [m]', fmt(r['b_w'], 4)),
        (r'$g_{\mathrm{span}}$ no ótimo', fmt(r['g_span'], 6)),
        (r'Restrições ativas', '%d' % r['n_ativas']),
    ]
    lines = [r'\begin{tabular}{lr}', r'\toprule',
             r'Grandeza & Valor \\', r'\midrule']
    for label, valor in itens:
        lines.append('%s & %s \\\\' % (label, valor))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_grandezas():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_grandezas.csv'),
                     index_col=0)
    labels = {
        'W0 [kgf]': r'$W_0$ [kgf]',
        'W_empty [kgf]': r'$W_e$ [kgf]',
        'W_fuel [kgf]': r'$W_f$ [kgf]',
        'T0 [kgf]': r'$T_0$ [kgf]',
        'T0req [kgf]': r'$T_{0,\mathrm{req}}$ [kgf]',
        'AR_w [-]': r'$AR_w$',
        'S_w [m2]': r'$S_w$ [m$^2$]',
        'b_w [m]': r'$b_w$ [m]',
        'cr_w [m]': r'$c_{r,w}$ [m]',
        'ct_w [m]': r'$c_{t,w}$ [m]',
        'S_h [m2]': r'$S_h$ [m$^2$]',
        'S_v [m2]': r'$S_v$ [m$^2$]',
        'deltaS_wlan [m2]': r'$\Delta S_{wlan}$ [m$^2$]',
    }
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r"Grandeza & Partida & Ótimo & $\Delta$ [\%] \\", r'\midrule']
    for key in labels:
        if key not in df.index:
            continue
        row = df.loc[key]
        lines.append('%s & %s & %s & %s \\\\' % (
            labels[key], fmt(row['inicial'], 3), fmt(row['otimo'], 3),
            fmt(row['variacao_pct'], 2)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_multistart():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'ver_multistart.csv'))
    lines = [r'\begin{tabular}{rrrrrr}', r'\toprule',
             r'\multicolumn{2}{c}{Partida} & \multicolumn{4}{c}{Convergiu para} \\',
             r'\cmidrule(lr){1-2}\cmidrule(lr){3-6}',
             r'$AR_w$ & $S_w$ & $AR_w$ & $S_w$ [m$^2$] & '
             r'$b_w$ [m] & $W_0$ [kgf] \\',
             r'\midrule']
    for _, row in df.iterrows():
        lines.append('%s & %s & %s & %s & %s & %s \\\\' % (
            fmt(row['AR_w_ini'], 2), fmt(row['S_w_ini'], 1),
            fmt(row['AR_w'], 5), fmt(row['S_w'], 5),
            fmt(row['b_w'], 5), fmt(row['W0_kgf'], 4)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_verificacao():
    '''
    Resumo das quatro evidências de que o ótimo é interior.
    '''
    df_ms = pd.read_csv(os.path.join(RESULTS_DIR, 'ver_multistart.csv'))
    df_sr = pd.read_csv(os.path.join(RESULTS_DIR, 'ver_sem_restricao.csv'))
    df_gr = pd.read_csv(os.path.join(RESULTS_DIR, 'ver_gradiente.csv'))
    g = df_gr.iloc[0]

    com = df_sr[df_sr['caso'] == 'com restrição'].iloc[0]
    sem = df_sr[df_sr['caso'] == 'sem restrição'].iloc[0]

    itens = [
        (r'Dispersão de $W_0$ entre 8 partidas [kgf]',
         '%.1e' % (df_ms['W0_kgf'].max() - df_ms['W0_kgf'].min())),
        (r'$W_0$ com a restrição de envergadura [kgf]',
         fmt(com['W0_kgf'], 4)),
        (r'$W_0$ sem a restrição de envergadura [kgf]',
         fmt(sem['W0_kgf'], 4)),
        (r'Diferença entre os dois [kgf]',
         '%.1e' % abs(com['W0_kgf'] - sem['W0_kgf'])),
        (r'Melhor $W_0$ \emph{sobre} a fronteira $b_w=30$ m [kgf]',
         fmt(g['W0_melhor_fronteira_kgf'], 4)),
        (r'Penalidade de ir para a fronteira [kgf]',
         fmt(g['penalidade_fronteira_kgf'], 4)),
        (r'$\partial W_0/\partial AR_w$ no ótimo [kgf]',
         fmt(g['dW0_dAR_kgf'], 5)),
        (r'$\partial W_0/\partial S_w$ no ótimo [kgf/m$^2$]',
         fmt(g['dW0_dS_kgf_m2'], 5)),
    ]
    lines = [r'\begin{tabular}{lr}', r'\toprule',
             r'Evidência & Valor \\', r'\midrule']
    for label, valor in itens:
        lines.append('%s & %s \\\\' % (label, valor))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# =========================================

if __name__ == '__main__':

    os.makedirs(TEX_DIR, exist_ok=True)

    write('tab_resultados.tex', tab_resultados())
    write('tab_corrida.tex', tab_corrida())
    write('tab_grandezas.tex', tab_grandezas())
    write('tab_multistart.tex', tab_multistart())
    write('tab_verificacao.tex', tab_verificacao())
