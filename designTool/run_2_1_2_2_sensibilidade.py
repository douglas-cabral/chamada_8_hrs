'''
INSTITUTO TECNOLOGICO DE AERONAUTICA
PRJ-23 - Aircraft Preliminary Design
Homework 01 - DOE analysis - Grupo NJ-0502

Atividades 2.1 e 2.2 - Tabela de sensibilidade relativa.

Calcula, por diferencas finitas progressivas, a sensibilidade relativa

        S_ij = (dY_i / Y_i*) / (dX_j / X_j*)

para a aeronave padrao (Fokker 100) e para a aeronave do grupo
('my_airplane'), com as perturbacoes definidas no enunciado.

Saidas geradas em resultados/:
  sens_fokker100.csv / sens_my_airplane.csv   - tabelas numericas
  sens_fokker100.tex / sens_my_airplane.tex   - tabelas em LaTeX
  sens_fokker100.png / sens_my_airplane.png   - mapa de calor das tabelas
'''

# IMPORTS
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from doe_common import (deg2rad, get_baseline, get_input, perturb, run_case,
                        run_baseline)

# =========================================

# SETUP

RESULTS_DIR = 'resultados'

# Perturbacoes do enunciado.
#   ('rel', f) -> X = X* * (1 + f)
#   ('abs', d) -> X = X* + d          (d ja nas unidades internas do designTool)
PERTURBATIONS = [
    ('S_w',          r'$S_w$',        ('rel', 0.02),          '+2%'),
    ('AR_w',         r'$AR_w$',       ('rel', 0.02),          '+2%'),
    ('sweep_w',      r'$\Lambda_w$',  ('rel', 0.02),          '+2%'),
    ('dihedral_w',   r'$\delta_w$',   ('abs', 2.0 * deg2rad), '+2 deg'),
    ('xr_w',         r'$x_{r,w}$',    ('rel', 0.02),          '+2%'),
    ('Cht',          r'$C_{h,t}$',    ('rel', 0.02),          '+2%'),
    ('Mach_cruise',  r'$M$',          ('abs', 0.02),          '+0.02'),
    ('range_cruise', r'$R$',          ('rel', 0.02),          '+2%'),
]

# Saidas pedidas na Atividade 2.1/2.2
OUTPUTS = ['W0', 'W_f', 'W_e', 'SM_fwd', 'SM_aft']

OUTPUT_LABELS = {
    'W0':     r'$\Delta W_0/W_0^*$',
    'W_f':    r'$\Delta W_f/W_f^*$',
    'W_e':    r'$\Delta W_e/W_e^*$',
    'SM_fwd': r'$\Delta SM_{fwd}/SM_{fwd}^*$',
    'SM_aft': r'$\Delta SM_{aft}/SM_{aft}^*$',
}

# =========================================

# FUNCOES


def relative_sensitivity_table(baseline_name):
    '''
    Monta a tabela de sensibilidade relativa da aeronave indicada.
    Retorna (DataFrame de sensibilidades, dicionario com as saidas de
    referencia, DataFrame com o passo relativo aplicado a cada entrada).
    '''
    base_out = run_baseline(baseline_name)
    if base_out is None:
        raise RuntimeError('A analise da aeronave de referencia falhou: '
                           + baseline_name)

    rows = {}
    steps = {}

    for key, label, (mode, amount), _text in PERTURBATIONS:

        x_ref = get_input(get_baseline(baseline_name), key)

        if mode == 'rel':
            x_new = x_ref * (1.0 + amount)
        else:
            x_new = x_ref + amount

        # Passo relativo efetivamente aplicado: dX/X*
        dx_rel = (x_new - x_ref) / x_ref

        pert_out = run_case(perturb(baseline_name, key, x_new))
        if pert_out is None:
            rows[label] = {OUTPUT_LABELS[o]: np.nan for o in OUTPUTS}
            steps[label] = dx_rel
            continue

        row = {}
        for o in OUTPUTS:
            dy_rel = (pert_out[o] - base_out[o]) / base_out[o]
            row[OUTPUT_LABELS[o]] = dy_rel / dx_rel
        rows[label] = row
        steps[label] = dx_rel

    df = pd.DataFrame(rows).T[[OUTPUT_LABELS[o] for o in OUTPUTS]]
    df_steps = pd.Series(steps, name='dX/X*')

    return df, base_out, df_steps


def save_latex(df, path, caption, label):
    '''
    Exporta a tabela no layout pedido pelo enunciado.
    '''
    body = []
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            cells.append('---' if not np.isfinite(v) else '%.4f' % v)
        body.append('%s & %s \\\\' % (idx, ' & '.join(cells)))

    header = ' & '.join([r'\multicolumn{1}{c}{\shortstack{%s \\ $\Delta(\cdot)/(\cdot)^*$}}'
                         % c for c in df.columns])

    tex = '\n'.join([
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{%s}' % caption,
        r'\label{%s}' % label,
        r'\small',
        r'\begin{tabular}{l|ccccc}',
        r'\toprule',
        r'Input & %s \\' % header,
        r'\midrule',
        '\n'.join(body),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open(path, 'w') as f:
        f.write(tex)


def save_heatmap(df, path, title):
    '''
    Mapa de calor da tabela de sensibilidade (leitura visual rapida do que e
    relevante). A escala e simetrica e saturada em um valor robusto para que
    poucas celulas muito grandes nao apaguem o restante da tabela.
    '''
    data = df.values.astype(float)
    finite = np.abs(data[np.isfinite(data)])
    vmax = np.percentile(finite, 90) if finite.size else 1.0
    vmax = max(vmax, 1e-3)

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            txt = '---' if not np.isfinite(v) else '%.3f' % v
            shade = 'white' if np.isfinite(v) and abs(v) > 0.65 * vmax else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=9, color=shade)

    ax.set_title(title, fontsize=12, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('sensibilidade relativa (saturada em P90)', fontsize=9)
    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=1.2)
    ax.tick_params(which='minor', length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def step_study(baseline_name, steps=(0.005, 0.01, 0.02, 0.05)):
    '''
    Verificacao de consistencia do codigo (Sec. 1 do enunciado): a
    sensibilidade relativa so tem significado se for aproximadamente
    independente do tamanho do passo. Aqui repetimos o calculo de
    dW_0/W_0* / (dS_w/S_w*) e analogos para varios passos.

    Um resultado que oscila com o passo indica ruido numerico dos lacos de
    ponto fixo (tolerancia de 10 N em `weight` e `thrust_matching`); um
    resultado que deriva monotonicamente indica nao-linearidade.
    '''
    base_out = run_baseline(baseline_name)
    keys = [k for k, _lb, (mode, _a), _t in PERTURBATIONS if mode == 'rel']
    labels = [lb for k, lb, (mode, _a), _t in PERTURBATIONS if mode == 'rel']

    rows = {}
    for key, label in zip(keys, labels):
        x_ref = get_input(get_baseline(baseline_name), key)
        row = {}
        for h in steps:
            out = run_case(perturb(baseline_name, key, x_ref * (1.0 + h)))
            if out is None:
                row['%.1f%%' % (100 * h)] = np.nan
            else:
                row['%.1f%%' % (100 * h)] = \
                    ((out['W0'] - base_out['W0']) / base_out['W0']) / h
        rows[label] = row

    return pd.DataFrame(rows).T


# =========================================

# EXECUCAO

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cases = [
        ('fokker100',   'Aeronave padrão (Fokker 100)'),
        ('my_airplane', 'Aeronave do grupo NJ-0502'),
    ]

    for name, title in cases:

        df, base_out, df_steps = relative_sensitivity_table(name)

        print('\n' + '=' * 78)
        print('  %s' % title)
        print('=' * 78)
        print('  Valores de referencia:')
        print('    W0     = %10.1f kgf' % base_out['W0'])
        print('    W_f    = %10.1f kgf' % base_out['W_f'])
        print('    W_e    = %10.1f kgf' % base_out['W_e'])
        print('    SM_fwd = %10.4f'     % base_out['SM_fwd'])
        print('    SM_aft = %10.4f'     % base_out['SM_aft'])
        print('-' * 78)
        print(df.to_string(float_format=lambda v: '%9.4f' % v))
        print('-' * 78)
        print('  Passo relativo aplicado (dX/X*):')
        print(df_steps.to_string(float_format=lambda v: '%8.5f' % v))

        df.to_csv(os.path.join(RESULTS_DIR, 'sens_%s.csv' % name))
        save_latex(df,
                   os.path.join(RESULTS_DIR, 'sens_%s.tex' % name),
                   'Sensibilidade relativa --- %s.' % title,
                   'tab:sens_%s' % name)
        save_heatmap(df,
                     os.path.join(RESULTS_DIR, 'sens_%s.png' % name),
                     'Sensibilidade relativa — %s' % title)

    # Verificacao de consistencia: independencia do passo (aeronave do grupo)
    df_step = step_study('my_airplane')
    print('\n' + '=' * 78)
    print('  VERIFICACAO DE CONSISTENCIA - sensibilidade de W_0 vs. passo')
    print('  (aeronave do grupo; entradas com perturbacao relativa)')
    print('=' * 78)
    print(df_step.to_string(float_format=lambda v: '%9.4f' % v))
    df_step.to_csv(os.path.join(RESULTS_DIR, 'sens_passo_my_airplane.csv'))

    print('\nArquivos gravados em %s/' % RESULTS_DIR)
