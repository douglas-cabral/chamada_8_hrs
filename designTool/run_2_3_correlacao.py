'''
Homework 01 - DOE analysis - Grupo NJ-0502

Atividade 2.3 - Grafico de correlacao da aeronave do grupo.

Estrutura baseada no script `doe2.py`.
'''

# IMPORTS
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS

from aux_tools_doe import corrdot
from doe_common import deg2rad, get_baseline, perturb, run_case, set_input

# =========================================

# SETUP

RESULTS_DIR = 'resultados'
BASELINE = 'my_airplane'
SEED = 123
SAMPLE_SIZES = [40, 400]

# Plot type (0-simple, 1-complete) - o enunciado pede grafico legivel, entao
# usamos o completo (dispersao + histograma + coeficiente de Pearson).
PLOT_TYPE = 1

_base_inputs = get_baseline(BASELINE)['inputs']

# Variaveis de entrada: (chave no designTool, rotulo, limite inf, limite sup,
#                        fator de conversao para o rotulo)
INPUT_VARS = [
    ('S_w',        r'$S_w$ [m$^2$]',       0.90 * _base_inputs['S_w'],
                                           1.10 * _base_inputs['S_w'],   1.0),
    ('AR_w',       r'$AR_w$',              6.0,  14.0,                    1.0),
    ('sweep_w',    r'$\Lambda_w$ [deg]',   10.0, 40.0,                    deg2rad),
    ('dihedral_w', r'$\delta_w$ [deg]',    0.0,  5.0,                     deg2rad),
    ('xr_w',       r'$x_{r,w}$ [m]',       0.90 * _base_inputs['xr_w'],
                                           1.10 * _base_inputs['xr_w'],  1.0),
]

# Variaveis de saida: (chave, rotulo, divisor aplicado ao valor)
OUTPUT_VARS = [
    ('W0',     r'$W_0$ [$10^3$ kgf]', 1.0e3),
    ('W_f',    r'$W_f$ [$10^3$ kgf]', 1.0e3),
    ('SM_aft', r'$SM_{aft}$',         1.0),
]

# =========================================

# FUNCOES


def sample_and_evaluate(n_samples, seed=SEED):
    '''
    Sorteia n_samples pontos por LHS, roda `analyze` em cada um e devolve
      - DataFrame com entradas e saidas das amostras validas;
      - DataFrame com TODAS as amostras sorteadas e a flag de convergencia (usado para mapear a fronteira de viabilidade).
    '''
    lb = [v[2] for v in INPUT_VARS]
    ub = [v[3] for v in INPUT_VARS]

    np.random.seed(seed)
    problem = Problem(n_var=len(lb), xl=np.array(lb), xu=np.array(ub))
    X = LHS()(problem, n_samples).get('X')

    records = []
    converged = []

    for ii in range(n_samples):

        # Monta a aeronave da amostra a partir da referencia do grupo
        airplane = perturb(BASELINE, INPUT_VARS[0][0],
                           X[ii, 0] * INPUT_VARS[0][4])
        for jj in range(1, len(INPUT_VARS)):
            key, _label, _lo, _hi, factor = INPUT_VARS[jj]
            set_input(airplane, key, X[ii, jj] * factor)

        out = run_case(airplane)
        converged.append(out is not None)

        if out is None:
            continue

        rec = {}
        for jj, (_key, label, _lo, _hi, _f) in enumerate(INPUT_VARS):
            rec[label] = X[ii, jj]
        for key, label, divisor in OUTPUT_VARS:
            rec[label] = out[key] / divisor
        records.append(rec)

    df = pd.DataFrame(records)

    df_all = pd.DataFrame(X, columns=[v[1] for v in INPUT_VARS])
    df_all['convergiu'] = converged

    return df, df_all


def correlation_plot(df, df_all, path, title, plot_type=PLOT_TYPE):
    '''
    Matriz de correlacao no formato do script `doe2.py`

    Nos paineis de dispersao entrada x entrada do bloco 5x5 superior esquerdo,
    as amostras cujo dimensionamento divergiu aparecem como 'x' vermelhos.
    '''
    sns.set(style='white', font_scale=1.15)

    if plot_type == 0:
        grid = sns.pairplot(df, corner=True, height=1.9)
    else:
        grid = sns.PairGrid(df, diag_sharey=False, height=1.9)
        grid.map_lower(sns.regplot, lowess=True,
                       scatter_kws={'s': 14, 'alpha': 0.55},
                       line_kws={'color': 'black', 'linewidth': 1.8})
        grid.map_diag(sns.histplot, color='0.45')
        grid.map_upper(corrdot)

    # Sobrepoe as entradas dos casos divergentes somente nos scatter plots do
    # bloco entrada x entrada (triangulo inferior do quadrado 5x5). Os demais
    # paineis envolvem saidas inexistentes para esses casos e ficam inalterados.
    failed = df_all.loc[~df_all['convergiu'].astype(bool)]
    input_labels = [v[1] for v in INPUT_VARS]
    for row in range(1, len(input_labels)):
        for col in range(row):
            grid.axes[row, col].scatter(
                failed[input_labels[col]],
                failed[input_labels[row]],
                marker='x',
                s=38,
                color='#c53030',
                linewidths=1.4,
                alpha=0.9,
                zorder=6,
            )

    for ax in grid.axes.flatten():
        if ax is None:
            continue
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.tick_params(labelsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    grid.figure.suptitle(title, y=1.004, fontsize=25)
    grid.figure.tight_layout()
    grid.figure.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(grid.figure)


def feasibility_plot(df_all, path, title):
    '''
    Mapa das amostras que fecharam (ou nao) o laco de ponto fixo de `weight`, projetado no plano (Lambda_w, AR_w)
    '''
    sns.set(style='whitegrid', font_scale=1.0)
    lam = df_all[INPUT_VARS[2][1]]
    ar = df_all[INPUT_VARS[1][1]]
    ok = df_all['convergiu'].values.astype(bool)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter(lam[ok], ar[ok], s=26, c='#2b6cb0', alpha=0.75,
               edgecolors='none', label='convergiu (%d)' % ok.sum())
    ax.scatter(lam[~ok], ar[~ok], s=32, c='#c53030', alpha=0.85,
               marker='x', linewidths=1.4,
               label='divergiu (%d)' % (~ok).sum())
    ax.set_xlabel(INPUT_VARS[2][1])
    ax.set_ylabel(INPUT_VARS[1][1])
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sweep_trend_plot(path):
    '''
    Varredura unidimensional em Lambda_w (demais entradas na referencia)
    '''
    from designTool.geometry import change_sweep, geometry

    sweep_deg = np.linspace(10.0, 45.0, 141)
    w0 = np.full_like(sweep_deg, np.nan)
    wf = np.full_like(sweep_deg, np.nan)
    mcrit = np.zeros_like(sweep_deg)
    cdwave = np.zeros_like(sweep_deg)

    base = get_baseline(BASELINE)['inputs']
    tcm = 0.25 * base['tcr_w'] + 0.75 * base['tct_w']
    CL_ref = 0.5   # CL tipico de cruzeiro, apenas para ilustrar a equacao

    for i, deg in enumerate(sweep_deg):
        airplane = perturb(BASELINE, 'sweep_w', deg * deg2rad)
        geometry(airplane)
        out = run_case(airplane)

        g = airplane['geometry']
        inp = airplane['inputs']
        s50 = change_sweep(0.25, 0.50, inp['sweep_w'],
                           g['b_w'] / 2, g['cr_w'], g['ct_w'])
        # Korn (aerodynamics.py, l. 269-272)
        m_dd = (inp['k_korn'] / np.cos(s50)
                - tcm / np.cos(s50) ** 2
                - CL_ref / 10 / np.cos(s50) ** 3)
        mcrit[i] = m_dd - (0.1 / 80) ** (1 / 3)
        cdwave[i] = 20 * max(0.0, base['Mach_cruise'] - mcrit[i]) ** 4

        if out is not None:
            w0[i] = out['W0'] / 1.0e3
            wf[i] = out['W_f'] / 1.0e3

    sns.set(style='whitegrid', font_scale=1.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=True)

    ax1.plot(sweep_deg, cdwave, color='#c53030', linewidth=2,
             label=r'$C_{D,wave}$ em $M=0{,}85$')
    ax1.set_ylabel(r'$C_{D,wave}$', color='#c53030')
    ax1.tick_params(axis='y', labelcolor='#c53030')
    ax1b = ax1.twinx()
    ax1b.plot(sweep_deg, mcrit, color='#2b6cb0', linewidth=2, linestyle='--',
              label=r'$M_{crit}$ (Korn)')
    ax1b.axhline(base['Mach_cruise'], color='0.35', linestyle=':',
                 linewidth=1.5)
    ax1b.annotate(r'$M_{cruise} = 0{,}85$',
                  xy=(31.0, base['Mach_cruise']),
                  textcoords='offset points', xytext=(0, -14),
                  fontsize=9.5, color='0.30',
                  bbox=dict(boxstyle='round,pad=0.22', fc='white',
                            ec='none', alpha=0.85))
    ax1b.set_ylabel(r'$M_{crit}$', color='#2b6cb0')
    ax1b.tick_params(axis='y', labelcolor='#2b6cb0')
    ax1b.grid(False)
    ax1.set_title(r'Origem da correlação $\Lambda_w \to W_0$: arrasto de onda',
                  fontsize=12)

    ax2.plot(sweep_deg, w0, color='#2b6cb0', linewidth=2, label=r'$W_0$')
    ax2.plot(sweep_deg, wf, color='#dd6b20', linewidth=2, label=r'$W_f$')

    finite = np.isfinite(w0)
    if finite.any():
        k = int(np.nanargmin(w0))
        ax2.scatter([sweep_deg[k]], [w0[k]], s=90, marker='*', zorder=5,
                    color='#2b6cb0', edgecolors='k', linewidths=0.6)
        ax2.annotate(r'$W_0$ mínimo em $\Lambda_w \approx %.0f^\circ$'
                     % sweep_deg[k],
                     (sweep_deg[k], w0[k]), textcoords='offset points',
                     xytext=(10, 18), fontsize=10)

        lim = sweep_deg[finite].min()
        ax2.axvspan(sweep_deg[0], lim, color='#c53030', alpha=0.12)
        ax2.text(sweep_deg[0] + 0.4, np.nanmax(w0) * 0.97,
                 'dimensionamento\ndiverge', fontsize=9, color='#c53030',
                 va='top')

    ax2.set_xlabel(r'$\Lambda_w$ [deg]')
    ax2.set_ylabel(r'peso [$10^3$ kgf]')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title('Resposta do dimensionamento (demais entradas na '
                  'referência)', fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# =========================================

# EXECUCAO

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for n_samples in SAMPLE_SIZES:

        df, df_all = sample_and_evaluate(n_samples)
        n_fail = int((~df_all['convergiu']).sum())

        print('\n' + '=' * 78)
        print('  DOE com %d amostras (LHS, seed=%d)' % (n_samples, SEED))
        print('=' * 78)
        print('  amostras validas : %d' % len(df))
        print('  amostras perdidas: %d (%.1f%%)'
              % (n_fail, 100.0 * n_fail / n_samples))
        print('-' * 78)
        print(df.describe().to_string(float_format=lambda v: '%12.4f' % v))

        df.to_csv(os.path.join(RESULTS_DIR, 'amostras_%03d.csv' % n_samples),
                  index=False)
        df_all.to_csv(os.path.join(RESULTS_DIR,
                                   'viabilidade_%03d.csv' % n_samples),
                      index=False)

        correlation_plot(
            df,
            df_all,
            os.path.join(RESULTS_DIR, 'correlacao_%03d.png' % n_samples),
            'Gráfico de correlação — aeronave do grupo NJ-0502 '
            '(%d amostras, LHS)' % n_samples)

        feasibility_plot(
            df_all,
            os.path.join(RESULTS_DIR, 'viabilidade_%03d.png' % n_samples),
            'Convergência do dimensionamento — %d amostras '
            '($M_{cruise} = 0{,}85$)' % n_samples)

        if n_samples == max(SAMPLE_SIZES):
            corr = df.corr('pearson')
            print('-' * 78)
            print('  Matriz de correlacao de Pearson:')
            print(corr.to_string(float_format=lambda v: '%7.3f' % v))
            corr.to_csv(os.path.join(RESULTS_DIR,
                                     'correlacoes_%03d.csv' % n_samples))

    # Varredura 1-D de apoio a discussao das tendencias
    sweep_trend_plot(os.path.join(RESULTS_DIR, 'tendencia_sweep.png'))

    print('\nArquivos gravados em %s/' % RESULTS_DIR)
