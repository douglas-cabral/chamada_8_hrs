'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 2 - Grupo NJ-0502

Etapa 1 - Minimização do MTOW da aeronave padrão (Fokker 100) com SLSQP.

Problema:
    min   W0(AR_w, S_w)
    s.a.  b_w <= 30 m
          7 <= AR_w <= 12
          80 <= S_w <= 120 m2
    partida: AR_w = 7,5   S_w = 90 m2

O roteiro manda deixar o otimizador estimar as derivadas por diferenças
finitas, então nenhum jacobiano é fornecido ao `minimize`.

Uso:  python run_2_1_otimizacao.py
'''

# IMPORTS
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opt_padrao import (B_W_MAX, CONSTRAINTS, DESIGN_VARS, DV_NAMES, Model,
                        constraint_vector, gravity, physical_report)

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_aeronave_padrao')

OPTIONS = {'maxiter': 200, 'ftol': 1e-8, 'disp': False}

# Folga numérica para classificar uma restrição como ativa no ótimo.
TOL_ATIVA = 1e-4

# =========================================


def run_opt(model=None):
    '''
    Roda o SLSQP sem jacobianos (diferenças finitas do próprio SciPy)
    e devolve o modelo, o resultado do scipy e o tempo de parede.
    '''
    if model is None:
        model = Model()

    cons = [{'type': 'ineq', 'fun': model.confun}]

    t_start = time.time()
    result = minimize(model.objfun, model.x0,
                      constraints=cons,
                      bounds=model.bounds,
                      method='slsqp',
                      options=OPTIONS)
    elapsed = time.time() - t_start

    return model, result, elapsed


def history_frame(model):
    '''
    Histórico das chamadas da função objetivo.

    O índice n_f = 1, 2, ... conta as vezes em que o SLSQP pediu f(x),
    incluindo as avaliações que o SciPy gasta montando as diferenças
    finitas (aqui elas passam pela mesma `objfun`).
    '''
    data = {'f': model.hist_f}

    x_hist = np.array(model.hist_x)
    for j, name in enumerate(model.dv_names):
        data[name] = x_hist[:, j]*model.scale[j]

    g_hist = np.array(model.hist_g)
    for j, spec in enumerate(CONSTRAINTS):
        data['g_' + spec[0]] = g_hist[:, j]

    data['b_w'] = (1.0 - g_hist[:, 0])*B_W_MAX

    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = 'n_f'
    return df


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print('=' * 78)
    print('  OTIMIZAÇÃO DA AERONAVE PADRÃO (Fokker 100)')
    print('  Variáveis: %s' % ', '.join(DV_NAMES))
    print('  Restrição: b_w <= %.1f m' % B_W_MAX)
    print('=' * 78)

    model, result, elapsed = run_opt()

    res = model.results(result.x)
    g = constraint_vector(res)
    x_phys = model.to_physical(result.x)
    ganho = 100.0*(1.0 - res['W0']/model.W0_ref)

    res0 = model.res0

    print('  status         : %s' % result.message)
    print('  iterações SLSQP: %d' % result.nit)
    print('  nfev (SciPy)   : %d' % result.nfev)
    print('  njev (SciPy)   : %d' % result.njev)
    print('  chamadas de f  : %d' % model.n_objfun)
    print('  chamadas de g  : %d' % model.n_confun)
    print('  aval. designTool: %d (pontos distintos; cache ativo)'
          % model.n_designTool)
    print('  tempo          : %.3f s' % elapsed)
    print('  W0             : %.2f kgf  (%.3f%% de redução)'
          % (res['W0']/gravity, ganho))
    print('  b_w            : %.4f m' % res['b_w'])
    print('  menor g        : %+.6f  (%s)'
          % (g.min(), 'viável' if g.min() >= -TOL_ATIVA else 'INVIÁVEL'))

    for name, value in zip(model.dv_names, x_phys):
        lo, hi = [spec for spec in DESIGN_VARS if spec[0] == name][0][4:6]
        marca = ''
        if abs(value - lo) < 1e-6*max(1.0, abs(lo)):
            marca = '  <- no limite inferior'
        elif abs(value - hi) < 1e-6*max(1.0, abs(hi)):
            marca = '  <- no limite superior'
        print('    %-6s %12.5f%s' % (name, value, marca))

    history_frame(model).to_csv(os.path.join(RESULTS_DIR, 'opt_hist.csv'))

    # ---------------------------------------------------------------
    # Tabela 1 do roteiro

    df_tab1 = pd.DataFrame([
        {'ponto': 'Partida',
         'AR_w': model.x0_phys[0], 'S_w': model.x0_phys[1],
         'MTOW_kgf': res0['W0']/gravity, 'MTOW_N': res0['W0'],
         'b_w': res0['b_w']},
        {'ponto': 'Ótimo',
         'AR_w': x_phys[0], 'S_w': x_phys[1],
         'MTOW_kgf': res['W0']/gravity, 'MTOW_N': res['W0'],
         'b_w': res['b_w']},
    ]).set_index('ponto')
    df_tab1.to_csv(os.path.join(RESULTS_DIR, 'opt_tabela1.csv'))

    # ---------------------------------------------------------------
    # Corrida

    pd.DataFrame([{
        'sucesso': bool(result.success),
        'status': result.message,
        'n_vars': len(model.dv_names),
        'n_iter': int(result.nit),
        'nfev_scipy': int(result.nfev),
        'njev_scipy': int(result.njev),
        'n_objfun': model.n_objfun,
        'n_confun': model.n_confun,
        'n_designTool': model.n_designTool,
        'tempo_s': elapsed,
        'W0_ini_kgf': res0['W0']/gravity,
        'W0_kgf': res['W0']/gravity,
        'f': res['W0']/model.W0_ref,
        'ganho_pct': ganho,
        'b_w': res['b_w'],
        'g_span': g[0],
        'g_min': g.min(),
        'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
    }]).to_csv(os.path.join(RESULTS_DIR, 'opt_corrida.csv'), index=False)

    # ---------------------------------------------------------------
    # Variáveis, restrições e grandezas

    rows_dv = []
    for name, x0v, xov in zip(DV_NAMES, model.x0_phys, x_phys):
        spec = [s for s in DESIGN_VARS if s[0] == name][0]
        rows_dv.append({'variavel': name,
                        'label': spec[1],
                        'unidade': spec[2],
                        'inicial': x0v,
                        'otimo': xov,
                        'lim_inf': spec[4],
                        'lim_sup': spec[5],
                        'variacao_pct': 100.0*(xov/x0v - 1.0),
                        'no_limite': bool(
                            abs(xov - spec[4]) < 1e-6*max(1.0, abs(spec[4]))
                            or abs(xov - spec[5]) < 1e-6*max(1.0, abs(spec[5])))})
    df_dv = pd.DataFrame(rows_dv).set_index('variavel')
    df_dv.to_csv(os.path.join(RESULTS_DIR, 'opt_variaveis.csv'))

    g_base = constraint_vector(res0)
    rows_con = []
    for spec, gb, go in zip(CONSTRAINTS, g_base, g):
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
    for (label, v_base), (_, v_opt) in zip(physical_report(res0),
                                           physical_report(res)):
        rows_phys.append({'grandeza': label,
                          'inicial': v_base,
                          'otimo': v_opt,
                          'variacao_pct': (100.0*(v_opt/v_base - 1.0)
                                           if abs(v_base) > 1e-12 else np.nan)})
    df_phys = pd.DataFrame(rows_phys).set_index('grandeza')
    df_phys.to_csv(os.path.join(RESULTS_DIR, 'opt_grandezas.csv'))

    print('\n' + '=' * 78)
    print('  TABELA 1 DO ROTEIRO')
    print('=' * 78)
    print(df_tab1[['AR_w', 'S_w', 'MTOW_kgf', 'b_w']].to_string(
        float_format=lambda v: '%12.4f' % v))

    print('\n  Variáveis de projeto:')
    print(df_dv[['inicial', 'otimo', 'lim_inf', 'lim_sup',
                 'variacao_pct']].to_string(
        float_format=lambda v: '%10.4f' % v))

    print('\n  Restrições normalizadas:')
    print(df_con[['g_inicial', 'g_otimo', 'ativa']].to_string(
        float_format=lambda v: '%10.6f' % v))

    print('\n  Grandezas dimensionais:')
    print(df_phys.to_string(float_format=lambda v: '%12.4f' % v))

    ativas = df_con.index[df_con['ativa']].tolist()
    print('\n  Restrições ativas no ótimo: %s'
          % (', '.join(ativas) if ativas else 'nenhuma'))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
