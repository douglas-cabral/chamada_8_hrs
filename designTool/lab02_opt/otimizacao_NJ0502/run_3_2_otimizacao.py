'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 3 - Grupo NJ-0502

Etapa 2 - Otimização de MTOW com SLSQP.

Uma única corrida, com o conjunto completo de variáveis de projeto.
Parte da configuração de PRJ-22.

Uso:  python run_3_2_otimizacao.py
'''

# IMPORTS
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opt_common import (CONSTRAINTS, DESIGN_VARS, DV_NAMES, Model,
                        constraint_vector, gravity, physical_report)

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados')

OPTIONS = {'maxiter': 200, 'ftol': 1e-6, 'disp': False}

# Folga numérica para classificar uma restrição como ativa no ótimo
TOL_ATIVA = 1e-4

# =========================================


def run_opt(dv_names):
    '''
    Roda o SLSQP e devolve o modelo, o resultado do scipy e o tempo de parede.
    '''
    model = Model(dv_names)

    cons = [{'type': 'ineq',
             'fun': model.confun,
             'jac': model.conjac}]

    t_start = time.time()
    result = minimize(model.objfun, model.x0,
                      jac=model.objgrad,
                      constraints=cons,
                      bounds=model.bounds,
                      method='slsqp',
                      options=OPTIONS)
    elapsed = time.time() - t_start

    return model, result, elapsed


def history_frame(model):
    '''
    Histórico das chamadas da função objetivo, com variáveis e restrições.

    O índice n_f = 1, 2, ... conta as vezes em que o SLSQP pediu f(x).
    As avaliações usadas só nos gradientes (diferenças finitas) não entram.
    '''
    data = {'f': model.hist_f}

    x_hist = np.array(model.hist_x)
    for j, name in enumerate(model.dv_names):
        data[name] = x_hist[:, j]*model.scale[j]

    g_hist = np.array(model.hist_g)
    for j, spec in enumerate(CONSTRAINTS):
        data['g_' + spec[0]] = g_hist[:, j]

    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = 'n_f'
    return df


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 78)
    print('  OTIMIZAÇÃO  (%d variáveis)' % len(DV_NAMES))
    print('  Variáveis: %s' % ', '.join(DV_NAMES))
    print('=' * 78)

    model, result, elapsed = run_opt(DV_NAMES)

    res = model.results(result.x)
    g = constraint_vector(res)
    x_phys = model.to_physical(result.x)
    ganho = 100.0*(1.0 - res['W0']/model.W0_ref)

    print('  status        : %s' % result.message)
    print('  iterações SLSQP: %d' % result.nit)
    print('  chamadas de f : %d  (eixo n_f das figuras)' % model.n_objfun)
    n = len(DV_NAMES)
    print('  aval. modelo  : %d  (1+2n=%d por chamada de f)'
          % (model.n_designTool, 1 + 2*n))
    print('  tempo         : %.2f s' % elapsed)
    print('  W0            : %.1f kgf  (%.2f%% de redução)'
          % (res['W0']/gravity, ganho))
    print('  menor g       : %+.5f  (%s)'
          % (g.min(), 'viável' if g.min() >= -TOL_ATIVA else 'INVIÁVEL'))

    for name, value in zip(model.dv_names, x_phys):
        lo, hi = [spec for spec in DESIGN_VARS if spec[0] == name][0][4:6]
        marca = ''
        if abs(value - lo) < 1e-6*max(1.0, abs(lo)):
            marca = '  <- no limite inferior'
        elif abs(value - hi) < 1e-6*max(1.0, abs(hi)):
            marca = '  <- no limite superior'
        print('    %-8s %12.5f%s' % (name, value, marca))

    history_frame(model).to_csv(os.path.join(RESULTS_DIR, 'opt_hist.csv'))

    pd.DataFrame([{
        'sucesso': bool(result.success),
        'status': result.message,
        'n_vars': len(model.dv_names),
        'n_iter': int(result.nit),
        'n_objfun': model.n_objfun,
        'n_designTool': model.n_designTool,
        'tempo_s': elapsed,
        'W0_kgf': res['W0']/gravity,
        'f': res['W0']/model.W0_ref,
        'ganho_pct': ganho,
        'g_min': g.min(),
        'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
    }]).to_csv(os.path.join(RESULTS_DIR, 'opt_corrida.csv'), index=False)

    # ---------------------------------------------------------------
    # Ótimo x configuração de partida

    x_opt = x_phys / model.scale
    res_base = model.res0
    res_opt = res
    g_base = constraint_vector(res_base)
    g_opt = g

    rows_dv = []
    for name, x0v, xov in zip(DV_NAMES,
                              model.to_physical(model.x0),
                              x_phys):
        spec = [s for s in DESIGN_VARS if s[0] == name][0]
        fator = spec[3]
        rows_dv.append({'variavel': name,
                        'label': spec[1],
                        'unidade': spec[2],
                        'inicial': x0v*fator,
                        'otimo': xov*fator,
                        'lim_inf': spec[4]*fator,
                        'lim_sup': spec[5]*fator,
                        'variacao_pct': 100.0*(xov/x0v - 1.0),
                        'inicial_si': x0v,
                        'otimo_si': xov})
    df_dv = pd.DataFrame(rows_dv).set_index('variavel')
    df_dv.to_csv(os.path.join(RESULTS_DIR, 'opt_variaveis.csv'))

    rows_con = []
    for spec, gb, go in zip(CONSTRAINTS, g_base, g_opt):
        rows_con.append({'restricao': spec[0],
                         'descricao': spec[1],
                         'expressao': spec[2],
                         'origem': spec[4],
                         'g_inicial': gb,
                         'g_otimo': go,
                         'ativa': bool(abs(go) <= TOL_ATIVA)})
    df_con = pd.DataFrame(rows_con).set_index('restricao')
    df_con.to_csv(os.path.join(RESULTS_DIR, 'opt_restricoes.csv'))

    rows_phys = []
    for (label, v_base), (_, v_opt) in zip(physical_report(res_base),
                                           physical_report(res_opt)):
        rows_phys.append({'grandeza': label,
                          'inicial': v_base,
                          'otimo': v_opt,
                          'variacao_pct': (100.0*(v_opt/v_base - 1.0)
                                           if abs(v_base) > 1e-12 else np.nan)})
    df_phys = pd.DataFrame(rows_phys).set_index('grandeza')
    df_phys.to_csv(os.path.join(RESULTS_DIR, 'opt_grandezas.csv'))

    print('\n' + '=' * 78)
    print('  ÓTIMO x CONFIGURAÇÃO DE PARTIDA')
    print('=' * 78)
    print('\n  Variáveis de projeto:')
    print(df_dv[['inicial', 'otimo', 'lim_inf', 'lim_sup',
                 'variacao_pct']].to_string(
        float_format=lambda v: '%10.4f' % v))

    print('\n  Restrições normalizadas:')
    print(df_con[['g_inicial', 'g_otimo', 'ativa']].to_string(
        float_format=lambda v: '%10.5f' % v))

    print('\n  Grandezas dimensionais:')
    print(df_phys.to_string(float_format=lambda v: '%12.4f' % v))

    ativas = df_con.index[df_con['ativa']].tolist()
    print('\n  Restrições ativas no ótimo: %s'
          % (', '.join(ativas) if ativas else 'nenhuma'))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
