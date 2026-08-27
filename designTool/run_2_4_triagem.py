'''
Homework 01 - DOE analysis - Grupo NJ-0502

Atividade 2.4 - Triagem das variaveis de projeto.
'''

# IMPORTS
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from doe_common import get_baseline, get_input, perturb, run_baseline, run_case

# =========================================

# SETUP

RESULTS_DIR = 'resultados'
BASELINE = 'my_airplane'

# Passo relativo das diferencas centradas
H_REL = 0.02

# |S| a partir do qual consideramos que a variavel "mexe" na saida
LIMIAR = 0.10

# Saidas que viram requisito/restricao na proxima etapa (Sec. 2.4)
OUTPUTS = [
    ('W0',               r'$W_0$'),
    ('W_f',              r'$W_f$'),
    ('deltaS_wlan',      r'$\Delta S_{wlan}$'),
    ('SM_fwd',           r'$SM_{fwd}$'),
    ('SM_aft',           r'$SM_{aft}$'),
    ('CLv',              r'$C_{Lv}$'),
    ('frac_nlg_fwd',     r'$f_{nlg,fwd}$'),
    ('frac_nlg_aft',     r'$f_{nlg,aft}$'),
    ('alpha_tipback',    r'$\alpha_{tip}$'),
    ('alpha_tailstrike', r'$\alpha_{tail}$'),
    ('phi_overturn',     r'$\phi_{ovt}$'),
    ('V_maxfuel',        r'$V_{tank}$'),
]

# Inputs candidatos
INPUTS = [
    # Asa
    ('S_w',              r'$S_w$'),
    ('AR_w',             r'$AR_w$'),
    ('taper_w',          r'$\lambda_w$'),
    ('sweep_w',          r'$\Lambda_w$'),
    ('dihedral_w',       r'$\delta_w$'),
    ('xr_w',             r'$x_{r,w}$'),
    ('zr_w',             r'$z_{r,w}$'),
    ('tcr_w',            r'$(t/c)_{r,w}$'),
    ('tct_w',            r'$(t/c)_{t,w}$'),
    # Empenagem horizontal
    ('Cht',              r'$C_{ht}$'),
    ('Lc_h',             r'$L_c/c_h$'),
    ('AR_h',             r'$AR_h$'),
    ('taper_h',          r'$\lambda_h$'),
    ('sweep_h',          r'$\Lambda_h$'),
    ('eta_h',            r'$\eta_h$'),
    ('zr_h',             r'$z_{r,h}$'),
    # Empenagem vertical
    ('Cvt',              r'$C_{vt}$'),
    ('Lb_v',             r'$L_b/b_v$'),
    ('AR_v',             r'$AR_v$'),
    # Fuselagem e nacele
    ('L_f',              r'$L_f$'),
    ('D_f',              r'$D_f$'),
    ('x_n',              r'$x_n$'),
    ('y_n',              r'$y_n$'),
    ('L_n',              r'$L_n$'),
    ('D_n',              r'$D_n$'),
    # Trem de pouso
    ('x_nlg',            r'$x_{nlg}$'),
    ('x_mlg',            r'$x_{mlg}$'),
    ('y_mlg',            r'$y_{mlg}$'),
    ('z_lg',             r'$z_{lg}$'),
    ('x_tailstrike',     r'$x_{ts}$'),
    ('z_tailstrike',     r'$z_{ts}$'),
    # Tanque
    ('c_tank_c_w',       r'$c_{tank}/c_w$'),
    ('x_tank_c_w',       r'$x_{tank}/c_w$'),
    ('b_tank_b_w_end',   r'$b_{tank}/b_w$'),
    # Hipersustentadores e perfil
    ('clmax_w',          r'$c_{l,max,w}$'),
    ('k_korn',           r'$k_{korn}$'),
    ('c_flap_c_wing',    r'$c_{flap}/c_w$'),
    ('b_flap_b_wing',    r'$b_{flap}/b_w$'),
    ('c_slat_c_wing',    r'$c_{slat}/c_w$'),
    ('b_slat_b_wing',    r'$b_{slat}/b_w$'),
    # Missao e propulsao
    ('Mach_cruise',      r'$M_{cr}$'),
    ('altitude_cruise',  r'$h_{cr}$'),
    ('range_cruise',     r'$R$'),
    ('distance_takeoff', r'$d_{TO}$'),
    ('distance_landing', r'$d_{LDG}$'),
    ('MLW_frac',         r'$MLW/MTOW$'),
    ('engine.BPR',       r'$BPR$'),
    ('engine.C_ref',     r'$C_{ref}$'),
    ('engine.Tmax',      r'$T_{max}$'),
    # Carga paga
    ('W_payload',        r'$W_{pay}$'),
    ('xcg_payload',      r'$x_{cg,pay}$'),
]

# =========================================

# FUNCOES


def screening_matrix(baseline_name, h_rel=H_REL):
    '''
    Constroi a matriz de sensibilidade relativa S_ij por diferencas centradas.
    '''
    base_out = run_baseline(baseline_name)
    if base_out is None:
        raise RuntimeError('A analise da aeronave de referencia falhou.')

    out_keys = [k for k, _ in OUTPUTS]
    out_labels = [lb for _, lb in OUTPUTS]

    rows = {}

    for key, label in INPUTS:

        x_ref = get_input(get_baseline(baseline_name), key)

        plus = run_case(perturb(baseline_name, key, x_ref * (1.0 + h_rel)))
        minus = run_case(perturb(baseline_name, key, x_ref * (1.0 - h_rel)))

        row = {}
        for k, lb in zip(out_keys, out_labels):
            if plus is None or minus is None or base_out[k] == 0.0:
                row[lb] = np.nan
            else:
                dy_rel = (plus[k] - minus[k]) / base_out[k]
                row[lb] = dy_rel / (2.0 * h_rel)
        rows[label] = row

    df = pd.DataFrame(rows).T[out_labels]

    return df, base_out


def rank_inputs(df, limiar=LIMIAR):
    '''
    Ranqueia as variaveis por intensidade (maior |S|) e abrangencia (numero de saidas afetadas acima do limiar)
    '''
    absdf = df.abs()
    ranking = pd.DataFrame({
        'S_max': absdf.max(axis=1),
        'saida_dominante': absdf.idxmax(axis=1),
        'n_saidas_relevantes': (absdf >= limiar).sum(axis=1),
        'S_medio': absdf.mean(axis=1),
    })
    ranking = ranking.sort_values(['S_max'], ascending=False)
    return ranking


def plot_heatmap(df, path, title):
    '''
    Mapa de calor da matriz S_ij
    '''
    data = df.values.astype(float)
    finite = np.abs(data[np.isfinite(data)])
    vmax = np.percentile(finite, 92) if finite.size else 1.0
    vmax = max(vmax, 1e-3)

    fig, ax = plt.subplots(figsize=(11.0, 0.32 * df.shape[0] + 2.6))
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=11, rotation=35, ha='right')
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                txt = '--'
            elif abs(v) < 5e-4:
                txt = '0'
            elif abs(v) >= 100:
                txt = '%.0f' % v
            else:
                txt = '%.2f' % v
            shade = 'white' if np.isfinite(v) and abs(v) > 0.65 * vmax else '0.15'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color=shade)

    ax.set_title(title, fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('sensibilidade relativa $S_{ij}$ (saturada em P92)',
                   fontsize=9)
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=0.9)
    ax.tick_params(which='minor', length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def plot_ranking(ranking, path, title, n_show=20):
    '''
    Barras horizontais com o ranking de influencia (escala log em S_max)
    '''
    sel = ranking.head(n_show).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(sel) + 1.8))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sel)))
    ax.barh(range(len(sel)), sel['S_max'].values, color=colors)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels(sel.index, fontsize=11)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\max_i |S_{ij}|$  (sensibilidade relativa)')
    ax.set_title(title, fontsize=12)
    ax.grid(axis='x', alpha=0.35, which='both')

    for k, (val, dom, n) in enumerate(zip(sel['S_max'],
                                          sel['saida_dominante'],
                                          sel['n_saidas_relevantes'])):
        ax.text(val * 1.12, k, '%s  (%d saídas)' % (dom, n),
                va='center', fontsize=8.5, color='0.25')

    ax.set_xlim(right=ax.get_xlim()[1] * 3.0)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


# =========================================

# EXECUCAO

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, base_out = screening_matrix(BASELINE)
    ranking = rank_inputs(df)

    print('\n' + '=' * 100)
    print('  TRIAGEM DE VARIAVEIS - aeronave do grupo NJ-0502')
    print('  diferencas centradas, passo relativo +-%.0f%%' % (100 * H_REL))
    print('=' * 100)
    print('  Saidas de referencia:')
    for k, lb in OUTPUTS:
        print('    %-18s = %12.4f' % (k, base_out[k]))
    print('-' * 100)
    print('  RANKING (por maior |S| entre as saidas):')
    print(ranking.to_string(float_format=lambda v: '%10.4f' % v))
    print('-' * 100)

    irrelevantes = ranking[ranking['S_max'] < 1e-6]
    print('  Variaveis SEM efeito mensuravel em nenhuma saida (S_max < 1e-6):')
    print('    ' + ', '.join(irrelevantes.index) if len(irrelevantes)
          else '    (nenhuma)')

    df.to_csv(os.path.join(RESULTS_DIR, 'triagem_completa.csv'))
    ranking.to_csv(os.path.join(RESULTS_DIR, 'triagem_ranking.csv'))

    plot_heatmap(df,
                 os.path.join(RESULTS_DIR, 'triagem_heatmap.png'),
                 'Sensibilidade relativa das saídas de projeto '
                 '— aeronave do grupo NJ-0502')

    # Mapa de calor reduzido com as 18 variaveis de maior influencia
    top = ranking.head(18).index
    plot_heatmap(df.loc[top],
                 os.path.join(RESULTS_DIR, 'triagem_heatmap_top.png'),
                 'As 18 variáveis de maior influência sobre as saídas '
                 'da Sec. 2.4')

    plot_ranking(ranking,
                 os.path.join(RESULTS_DIR, 'triagem_ranking.png'),
                 'Ranking de influência sobre as saídas da Sec. 2.4')

    print('\nArquivos gravados em %s/' % RESULTS_DIR)
