'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 3 - Grupo NJ-0502

Etapa 3 - Estudos de remoção de restrições e de relaxamento de limites.

O roteiro sugere, após a otimização de referência, desligar restrições para
observar como o ótimo se desloca. Cada variante abaixo isola uma das
restrições ativas e mostra para onde o projeto escorrega quando ela some,
o que serve de verificação de que a restrição está de fato governando o
resultado.

Uso:  python run_3_3_variantes.py     (rodar depois de run_3_2_otimizacao.py)
'''

# IMPORTS
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opt_common import (CON_NAMES, CONSTRAINTS, DV_NAMES, Model,
                        constraint_vector, gravity, rad2deg)

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_NJ0502')

OPTIONS = {'maxiter': 300, 'ftol': 1e-9, 'disp': False}
TOL_ATIVA = 1e-4

# Variantes: (rótulo, restrições removidas, limites sobrescritos)
VARIANTES = [
    ('V0 referência',        [],          {}),
    ('V1 sem C_Lv',          ['CLv'],     {}),
    ('V2 sem envergadura',   ['span'],    {}),
    ('V3 sem tanque',        ['tank'],    {}),
    ('V4 Cht até 0,50',      [],          {'Cht': (0.50, 1.10)}),
    ('V5 sem trem-na-asa',   ['gear_te'], {}),
]

# Grandezas acompanhadas em cada variante
WATCH = ['S_w', 'AR_w', 'sweep_w', 'xr_w', 'Cht', 'Cvt',
         'x_mlg', 'y_mlg', 'z_lg']

# =========================================


def solve(con_removidas, bounds_phys):
    '''
    Roda a otimização completa com um subconjunto de restrições.
    '''
    con_names = [n for n in CON_NAMES if n not in con_removidas]

    model = Model(DV_NAMES, con_names=con_names, bounds_phys=bounds_phys)

    result = minimize(model.objfun, model.x0,
                      jac=model.objgrad,
                      constraints=[{'type': 'ineq',
                                    'fun': model.confun,
                                    'jac': model.conjac}],
                      bounds=model.bounds,
                      method='slsqp',
                      options=OPTIONS)

    return model, result


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []

    for rotulo, removidas, bounds_phys in VARIANTES:

        model, result = solve(removidas, bounds_phys)

        res = model.results(result.x)
        x_phys = model.to_physical(result.x)

        # Restrições: sempre reportamos o conjunto completo, inclusive as
        # que foram desligadas, para enxergar o quanto elas são violadas.
        g_todas = constraint_vector(res)

        row = {'variante': rotulo,
               'removidas': ', '.join(removidas) if removidas else '-',
               'W0_kgf': res['W0']/gravity,
               'ganho_pct': 100.0*(1.0 - res['W0']/model.W0_ref),
               'sucesso': bool(result.success)}

        row.update(dict(zip(WATCH, x_phys)))
        row['sweep_w'] = row['sweep_w']*rad2deg
        row['b_w'] = res['b_w']
        row['CLv'] = res['CLv']
        row['tank_excess'] = res['tank_excess']

        for spec, g in zip(CONSTRAINTS, g_todas):
            row['g_' + spec[0]] = g

        # Restrições ativas considerando só as que foram impostas
        ativas = [spec[0] for spec, g in zip(CONSTRAINTS, g_todas)
                  if spec[0] in model.con_names and abs(g) <= TOL_ATIVA]
        row['ativas'] = ', '.join(ativas)

        rows.append(row)

        print('=' * 78)
        print('  %s' % rotulo)
        print('=' * 78)
        print('  W0        : %.1f kgf  (%.2f%% de redução)'
              % (row['W0_kgf'], row['ganho_pct']))
        print('  S_w %.2f m2 | AR_w %.3f | sweep %.2f deg | b_w %.2f m'
              % (row['S_w'], row['AR_w'], row['sweep_w'], row['b_w']))
        print('  Cht %.4f | Cvt %.5f | CLv %.4f | tank_excess %+.5f'
              % (row['Cht'], row['Cvt'], row['CLv'], row['tank_excess']))
        print('  x_mlg %.3f m | y_mlg %.3f m | b_mlg %.2f m | z_lg %.3f m'
              % (row['x_mlg'], row['y_mlg'], 2.0*row['y_mlg'], row['z_lg']))
        print('  ativas    : %s' % (row['ativas'] or 'nenhuma'))

        violadas = [spec[0] for spec, g in zip(CONSTRAINTS, g_todas)
                    if g < -TOL_ATIVA]
        if violadas:
            print('  passa a violar: %s' % ', '.join(violadas))
        print('')

    df = pd.DataFrame(rows).set_index('variante')
    df.to_csv(os.path.join(RESULTS_DIR, 'opt_variantes.csv'))

    print('=' * 78)
    print('  RESUMO DAS VARIANTES')
    print('=' * 78)
    cols = ['W0_kgf', 'ganho_pct', 'S_w', 'AR_w', 'b_w', 'Cht', 'Cvt', 'CLv']
    print(df[cols].to_string(float_format=lambda v: '%10.4f' % v))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
