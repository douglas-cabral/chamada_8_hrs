'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 3 - Grupo NJ-0502

Etapa 4 - Figuras do relatório.

  1. histórico da otimização, em figuras separadas (variáveis, objetivo
     e restrições);
  2. superposição das vistas em planta e lateral da aeronave de partida
     e da aeronave otimizada.

Uso:  python run_3_4_figuras.py   (rodar depois de run_3_2_otimizacao.py)
'''

# IMPORTS
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd

from opt_common import (CONSTRAINTS, DESIGN_VARS, DV_NAMES, get_baseline,
                        run_designTool)

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_NJ0502')

COR_BASE = '#1f4e79'
COR_OPT = '#c53030'

XLABEL_NF = r'chamada $n_f$ da função objetivo'

# Cores distintas (tab10 só tem 10; com 19 restrições o ciclo se repetia).
CORES_G = [
    '#c53030', '#2b6cb0', '#2f855a', '#dd6b20', '#6b46c1',
    '#0d9488', '#b83280', '#1a365d', '#276749', '#9c4221',
    '#553c9a', '#234e52', '#744210', '#9b2c2c', '#2c5282',
    '#22543d', '#7b341e', '#4a5568', '#718096',
]

GRUPO_G = [
    ('desempenho e pátio',
     'opt_hist_g_desempenho.png',
     ['landing', 'tank', 'thrust', 'span', 'wheelspan', 'height', 'CLv',
      'vt_te', 'ht_te']),
    ('estabilidade e trem',
     'opt_hist_g_estabilidade.png',
     ['SM_fwd', 'SM_aft', 'SM_aft_max', 'nlg_fwd', 'nlg_aft',
      'tipback', 'tailstrike', 'overturn', 'gear_te', 'gear_spar']),
]


def _load_hist():
    return pd.read_csv(os.path.join(RESULTS_DIR, 'opt_hist.csv'),
                       index_col=0)


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def fig_variaveis(df, path):
    specs = {s[0]: s for s in DESIGN_VARS}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name in DV_NAMES:
        serie = df[name]
        ax.plot(df.index, serie/serie.iloc[0], 'o-', ms=5, lw=1.4,
                label=specs[name][1])
    ax.axhline(1.0, color='gray', lw=0.6)
    ax.set_ylabel(r'$x/x_{\mathrm{inicial}}$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico das variáveis de projeto')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_xticks(list(df.index))
    _savefig(fig, path)


def fig_objetivo(df, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(df.index, df['f'], 'o-', ms=6, lw=1.8, color=COR_OPT)
    ax.set_ylabel(r'$f = W_0/W_{0,\mathrm{ref}}$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico da função objetivo')
    ax.set_xticks(list(df.index))
    _savefig(fig, path)


def fig_restricoes(df, path, titulo, nomes, cor0):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    k = cor0
    for name in nomes:
        col = 'g_' + name
        if col not in df.columns:
            continue
        if df[col].std() < 1e-10 and abs(df[col].iloc[-1]) > 5e-2:
            continue
        ax.plot(df.index, df[col], 'o-', ms=5, lw=1.4,
                color=CORES_G[k % len(CORES_G)], label=name)
        k += 1
    ax.axhline(0.0, color='gray', lw=0.9)
    ax.set_ylabel(r'$g$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico das restrições --- %s' % titulo)
    ax.set_ylim(-0.25, 1.35)
    ax.set_xticks(list(df.index))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8,
              frameon=True)
    _savefig(fig, path)
    return k


# =========================================
# 3. VISTAS SUPERPOSTAS


# Estações da fuselagem usadas pelo designTool (plots.py), em fração do
# comprimento e do diâmetro.
_FUS_H = [0.0, 2.27/4.0, 3.56/4.0, 1.0, 1.0, 1.07/4.0]
_FUS_W = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]


def _fuselagem(inputs):
    L_f = inputs['L_f']
    D_f = inputs['D_f']
    x_ts = inputs['x_tailstrike']
    xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_ts/L_f, 1.0])
    return xx*L_f, np.array(_FUS_H)*D_f, np.array(_FUS_W)*D_f


def _nacele_cilindro(ax, x0, y0, length, diam, cor, n_est=11):
    '''
    Silhueta de nacele igual ao plot_geometry do designTool: cilindro de
    eixo x, seção circular, 11 estações e linha de centro.
    '''
    r = 0.5*diam
    x1 = x0 + length
    ax.plot([x0, x1, x1, x0, x0],
            [y0 - r, y0 - r, y0 + r, y0 + r, y0 - r],
            color=cor, lw=1.2, zorder=3)
    for x in np.linspace(x0, x1, n_est):
        ax.plot([x, x], [y0 - r, y0 + r], color=cor, lw=0.7, zorder=3)
    ax.plot([x0, x1], [y0, y0], color=cor, lw=0.8, zorder=3)


def _naceles_planta(ax, inp, cor):
    for sy in (1.0, -1.0):
        _nacele_cilindro(ax, inp['x_n'], sy*inp['y_n'],
                         inp['L_n'], inp['D_n'], cor)


def _naceles_lateral(ax, inp, cor):
    _nacele_cilindro(ax, inp['x_n'], inp['z_n'],
                     inp['L_n'], inp['D_n'], cor)


def _planta(ax, airplane, cor, rotulo, com_fuselagem):
    geo = airplane['geometry']
    inp = airplane['inputs']

    if com_fuselagem:
        xf, _, wf = _fuselagem(inp)
        ax.plot(xf, wf/2, color='0.4', lw=1.0)
        ax.plot(xf, -wf/2, color='0.4', lw=1.0)
        _naceles_planta(ax, inp, '#d97706')

    # Asa
    xw = [inp['xr_w'], geo['xt_w'], geo['xt_w'] + geo['ct_w'],
          inp['xr_w'] + geo['cr_w']]
    yw = [0.0, geo['yt_w'], geo['yt_w'], 0.0]
    ax.plot(xw + xw[::-1], yw + [-v for v in yw[::-1]], color=cor, lw=1.6,
            label=rotulo)

    # Empenagem horizontal
    xh = [geo['xr_h'], geo['xt_h'], geo['xt_h'] + geo['ct_h'],
          geo['xr_h'] + geo['cr_h']]
    yh = [0.0, geo['yt_h'], geo['yt_h'], 0.0]
    ax.plot(xh + xh[::-1], yh + [-v for v in yh[::-1]], color=cor, lw=1.2,
            ls='--')

    # Trem de pouso principal
    ax.plot([inp['x_mlg']]*2, [inp['y_mlg'], -inp['y_mlg']], 'o',
            color=cor, ms=4)


def _lateral(ax, airplane, cor, rotulo, com_fuselagem):
    geo = airplane['geometry']
    inp = airplane['inputs']

    if com_fuselagem:
        xf, hf, _ = _fuselagem(inp)
        x_ts = inp['x_tailstrike']
        # Acima da estação de tailstrike o designTool achata o topo
        desloc = np.where(xf > x_ts, (inp['D_f'] - hf)/2, 0.0)
        z_top = hf/2 + desloc
        z_bot = -hf/2 + desloc
        ax.plot(xf, z_top, color='0.4', lw=1.0)
        ax.plot(xf, z_bot, color='0.4', lw=1.0)
        # Fecha o cone de cauda na estação L_f.
        ax.plot([xf[-1], xf[-1]], [z_top[-1], z_bot[-1]],
                color='0.4', lw=1.0)
        _naceles_lateral(ax, inp, '#d97706')

    # Corda de raiz da asa
    ax.plot([inp['xr_w'], inp['xr_w'] + geo['cr_w']],
            [inp['zr_w']]*2, color=cor, lw=2.4, label=rotulo)

    # Corda de raiz da empenagem horizontal
    ax.plot([geo['xr_h'], geo['xr_h'] + geo['cr_h']],
            [inp['zr_h']]*2, color=cor, lw=1.8, ls='--')

    # Empenagem vertical
    xv = [geo['xr_v'], geo['xt_v'], geo['xt_v'] + geo['ct_v'],
          geo['xr_v'] + geo['cr_v'], geo['xr_v']]
    zv = [inp['zr_v'], geo['zt_v'], geo['zt_v'], inp['zr_v'], inp['zr_v']]
    ax.plot(xv, zv, color=cor, lw=1.4)

    # Trem de pouso e linha do solo
    ax.plot([inp['x_nlg'], inp['x_mlg']], [inp['z_lg']]*2, 'o',
            color=cor, ms=5)
    ax.plot([0.0, inp['L_f']], [inp['z_lg']]*2, color=cor, lw=0.7, ls=':')

    # Linha de tailstrike
    ang = np.arctan((inp['z_tailstrike'] - inp['z_lg'])
                    / (inp['x_tailstrike'] - inp['x_mlg']))
    ax.plot([inp['x_mlg'], inp['L_f']],
            [inp['z_lg'], inp['z_lg'] + (inp['L_f'] - inp['x_mlg'])*np.tan(ang)],
            color=cor, lw=0.7, ls='-.')


def _xmax_aeronave(airplane):
    '''Estação mais a ré: fuselagem ou bordo de fuga das empenagens.'''
    geo = airplane['geometry']
    inp = airplane['inputs']
    return max(inp['L_f'],
               geo['xr_v'] + geo['cr_v'],
               geo['xt_v'] + geo['ct_v'],
               geo['xr_h'] + geo['cr_h'],
               geo['xt_h'] + geo['ct_h'])


def vistas(inputs_base, inputs_opt, path):
    '''
    Superpõe as vistas em planta e lateral das duas configurações.

    A vista lateral usa aspect='equal' com adjustable='box' e limites
    explícitos até L_f: com datalim o matplotlib cortava o cone de cauda
    para caber a envergadura no mesmo quadro.
    '''
    ap_base = run_designTool(inputs_base)
    ap_opt = run_designTool(inputs_opt)

    fig, axes = plt.subplots(2, 1, figsize=(11.6, 8.2),
                             gridspec_kw={'height_ratios': [2.15, 1.0]})

    _planta(axes[0], ap_base, COR_BASE, 'PRJ-22 (partida)', True)
    _planta(axes[0], ap_opt, COR_OPT, 'otimizada', False)
    axes[0].set_ylabel('y [m]')
    axes[0].set_title('Vista em planta')
    axes[0].legend(loc='upper right', fontsize=9)

    _lateral(axes[1], ap_base, COR_BASE, 'PRJ-22 (partida)', True)
    _lateral(axes[1], ap_opt, COR_OPT, 'otimizada', False)
    axes[1].set_ylabel('z [m]')
    axes[1].set_xlabel('x [m]')
    axes[1].set_title('Vista lateral')
    axes[1].legend(loc='upper right', fontsize=9)

    pad = 1.8
    x_max = max(_xmax_aeronave(ap_base), _xmax_aeronave(ap_opt))
    b_max = max(ap_base['geometry']['b_w'], ap_opt['geometry']['b_w'])
    z_lo = min(ap_base['inputs']['z_lg'], ap_opt['inputs']['z_lg'])
    z_hi = max(ap_base['geometry']['zt_v'], ap_opt['geometry']['zt_v'])

    axes[0].set_xlim(-pad, x_max + pad)
    axes[0].set_ylim(-b_max/2.0 - pad, b_max/2.0 + pad)
    axes[1].set_xlim(-pad, x_max + pad)
    axes[1].set_ylim(z_lo - pad, z_hi + pad)

    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='0.9', lw=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_dv = pd.read_csv(os.path.join(RESULTS_DIR, 'opt_variaveis.csv'),
                        index_col=0)

    inputs_base = get_baseline()
    inputs_opt = get_baseline()
    for name, row in df_dv.iterrows():
        inputs_opt[name] = row['otimo_si']

    vistas(inputs_base, inputs_opt,
           os.path.join(RESULTS_DIR, 'opt_vistas.png'))
    print('  gravado: opt_vistas.png')

    df_hist = _load_hist()
    fig_variaveis(df_hist, os.path.join(RESULTS_DIR, 'opt_hist_variaveis.png'))
    fig_objetivo(df_hist, os.path.join(RESULTS_DIR, 'opt_hist_objetivo.png'))
    k = 0
    for titulo, fname, nomes in GRUPO_G:
        k = fig_restricoes(df_hist,
                           os.path.join(RESULTS_DIR, fname),
                           titulo, nomes, k)

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
