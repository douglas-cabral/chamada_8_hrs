'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 3 - Grupo NJ-0502

Etapa 1 - Diagnóstico da configuração de partida.

Responde a duas perguntas antes de otimizar qualquer coisa:

  (a) a aeronave herdada de PRJ-22 satisfaz as restrições do enunciado?
  (b) o conjunto de variáveis de projeto e de restrições é suficiente?

O critério de suficiência é o posto da matriz de sensibilidades: uma
restrição cuja linha é identicamente nula não pode ser controlada por
nenhuma variável de projeto, e uma variável cuja coluna do objetivo é nula
só serve para viabilizar restrições.

Uso:  python run_3_1_diagnostico.py
'''

# IMPORTS
import os

import numpy as np
import pandas as pd

from opt_common import (CONSTRAINTS, DESIGN_VARS, Model, constraint_vector,
                        gravity, physical_report, rad2deg)

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_NJ0502')

ALL_DVS = [spec[0] for spec in DESIGN_VARS]

# Sensibilidade considerada nula
TOL_NULA = 1e-8

# =========================================


def baseline_table(model):
    '''
    Monta a tabela de restrições da configuração de partida.
    '''
    res = model.res0
    g = constraint_vector(res)

    rows = []
    for spec, gj in zip(CONSTRAINTS, g):
        rows.append({'restricao': spec[0],
                     'descricao': spec[1],
                     'origem': spec[4],
                     'g': gj,
                     'status': 'ok' if gj >= 0.0 else 'VIOLADA'})

    return pd.DataFrame(rows).set_index('restricao')


def sensitivity_matrix(model):
    '''
    Jacobiano normalizado do objetivo e das restrições no ponto de partida.
    '''
    gradf, jacg = model._finite_differences(model.x0)

    index = ['f = W0/W0ref'] + [spec[0] for spec in CONSTRAINTS]
    data = np.vstack([gradf, jacg])

    return pd.DataFrame(data, index=index, columns=model.dv_names)


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    model = Model(ALL_DVS)

    print('=' * 78)
    print('  CONFIGURAÇÃO DE PARTIDA - NJ-0502 (PRJ-22)')
    print('=' * 78)
    for label, value in physical_report(model.res0):
        print('  %-22s %12.4f' % (label, value))

    df_base = baseline_table(model)
    print('\n' + '=' * 78)
    print('  RESTRIÇÕES NORMALIZADAS NA CONFIGURAÇÃO DE PARTIDA')
    print('  (g >= 0 significa restrição satisfeita)')
    print('=' * 78)
    print(df_base.to_string(float_format=lambda v: '%9.4f' % v))

    n_viol = int((df_base['g'] < 0).sum())
    print('\n  Restrições violadas na partida: %d' % n_viol)

    df_sens = sensitivity_matrix(model)
    print('\n' + '=' * 78)
    print('  SENSIBILIDADES NORMALIZADAS  dF/dx_n  NO PONTO DE PARTIDA')
    print('  (x_n = x/|x_ref|, portanto as colunas são comparáveis entre si)')
    print('=' * 78)
    print(df_sens.to_string(float_format=lambda v: '%10.5f' % v))

    # --- Suficiência das variáveis de projeto ---
    obj_row = df_sens.loc['f = W0/W0ref'].abs()
    dv_inertes = list(obj_row[obj_row < TOL_NULA].index)

    # --- Suficiência das restrições ---
    con_rows = df_sens.drop(index='f = W0/W0ref').abs()
    con_inertes = list(con_rows.index[(con_rows < TOL_NULA).all(axis=1)])

    print('\n' + '=' * 78)
    print('  DIAGNÓSTICO DE SUFICIÊNCIA')
    print('=' * 78)
    print('  Variáveis sem efeito sobre o objetivo (atuam só nas restrições):')
    print('    %s' % (', '.join(dv_inertes) if dv_inertes else 'nenhuma'))
    print('  Restrições que nenhuma variável de projeto consegue alterar:')
    print('    %s' % (', '.join(con_inertes) if con_inertes else 'nenhuma'))

    # Variável que só é limitada por restrições adicionadas por nós
    print('\n  Restrições adicionadas por nós (fora da lista do enunciado):')
    for spec in CONSTRAINTS:
        if spec[4] == 'adicionada':
            print('    %-10s %s' % (spec[0], spec[1]))

    df_base.to_csv(os.path.join(RESULTS_DIR, 'diag_baseline.csv'))
    df_sens.to_csv(os.path.join(RESULTS_DIR, 'diag_sensibilidades.csv'))

    print('\n  Chamadas ao designTool: %d' % model.n_designTool)
    print('  Arquivos gravados em %s/' % RESULTS_DIR)
