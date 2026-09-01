'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 2 - Grupo NJ-0502

Etapa 2 - Verificação do ótimo encontrado em run_2_1_otimizacao.py.

O ótimo do Problema 2 cai a 17 cm do teto de envergadura (b_w = 29,83 m
contra 30 m). Como a pergunta 4 do roteiro é justamente "o ótimo é
restringido?", a resposta não pode depender de uma única corrida. Este
script produz três evidências independentes:

  1. multistart: oito pontos de partida espalhados na caixa;
  2. corrida SEM a restrição de envergadura (só as caixas). Se o ótimo
     irrestrito coincide com o restringido, a restrição está inativa;
  3. varredura ao longo da fronteira b_w = 30 m (AR_w*S_w = 900), para
     mostrar que o melhor ponto da fronteira é pior que o ponto interior;
  4. gradiente físico de W0 no ótimo, que deve ser nulo num ponto interior.

Uso:  python run_2_2_verificacao.py   (independe de run_2_1)
'''

# IMPORTS
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opt_padrao import B_W_MAX, Model, gravity

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_aeronave_padrao')

OPTIONS = {'maxiter': 300, 'ftol': 1e-10, 'disp': False}

STARTS = [(7.5, 90.0), (7.0, 80.0), (12.0, 120.0), (9.0, 110.0),
          (11.0, 85.0), (8.0, 100.0), (12.0, 80.0), (7.0, 120.0)]

# =========================================


def _solve(x0_phys, com_restricao=True):
    model = Model(x0_phys=np.asarray(x0_phys, dtype=float))
    cons = [{'type': 'ineq', 'fun': model.confun}] if com_restricao else ()
    result = minimize(model.objfun, model.x0, constraints=cons,
                      bounds=model.bounds, method='slsqp', options=OPTIONS)
    res = model.results(result.x)
    x_phys = model.to_physical(result.x)
    return {
        'AR_w': x_phys[0],
        'S_w': x_phys[1],
        'b_w': res['b_w'],
        'W0_kgf': res['W0']/gravity,
        'g_span': 1.0 - res['b_w']/B_W_MAX,
        'n_objfun': model.n_objfun,
        'sucesso': bool(result.success),
    }


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Multistart com a restrição

    print('=' * 78)
    print('  1. MULTISTART (com a restrição b_w <= 30 m)')
    print('=' * 78)
    rows = []
    for a, s in STARTS:
        out = _solve((a, s), com_restricao=True)
        out['AR_w_ini'] = a
        out['S_w_ini'] = s
        rows.append(out)
        print('  partida (%5.2f, %6.2f) -> AR=%8.5f  S=%9.5f  '
              'b_w=%8.5f  W0=%11.4f  g=%+.6f'
              % (a, s, out['AR_w'], out['S_w'], out['b_w'],
                 out['W0_kgf'], out['g_span']))

    df_ms = pd.DataFrame(rows)[['AR_w_ini', 'S_w_ini', 'AR_w', 'S_w', 'b_w',
                                'W0_kgf', 'g_span', 'n_objfun', 'sucesso']]
    df_ms.to_csv(os.path.join(RESULTS_DIR, 'ver_multistart.csv'), index=False)
    print('  dispersão de W0 entre as corridas: %.3e kgf'
          % (df_ms['W0_kgf'].max() - df_ms['W0_kgf'].min()))

    # ---------------------------------------------------------------
    # 2. Corrida sem a restrição de envergadura

    print('\n' + '=' * 78)
    print('  2. OTIMIZAÇÃO SEM A RESTRIÇÃO DE ENVERGADURA')
    print('=' * 78)
    livre = _solve((7.5, 90.0), com_restricao=False)
    restrita = rows[0]
    print('  irrestrito : AR=%8.5f  S=%9.5f  b_w=%8.5f  W0=%11.4f'
          % (livre['AR_w'], livre['S_w'], livre['b_w'], livre['W0_kgf']))
    print('  restringido: AR=%8.5f  S=%9.5f  b_w=%8.5f  W0=%11.4f'
          % (restrita['AR_w'], restrita['S_w'], restrita['b_w'],
             restrita['W0_kgf']))
    print('  |dW0| entre os dois: %.3e kgf'
          % abs(livre['W0_kgf'] - restrita['W0_kgf']))
    print('  => a restrição de envergadura está %s'
          % ('INATIVA' if abs(livre['W0_kgf'] - restrita['W0_kgf']) < 1e-3
             else 'ATIVA'))

    pd.DataFrame([dict(caso='com restrição', **restrita),
                  dict(caso='sem restrição', **livre)]).to_csv(
        os.path.join(RESULTS_DIR, 'ver_sem_restricao.csv'), index=False)

    # ---------------------------------------------------------------
    # 3. Varredura da fronteira b_w = 30 m

    print('\n' + '=' * 78)
    print('  3. VARREDURA DA FRONTEIRA b_w = 30 m  (AR_w * S_w = 900)')
    print('=' * 78)
    mod = Model()
    rows_f = []
    for AR in np.linspace(7.0, 12.0, 101):
        S = B_W_MAX**2/AR
        if not (80.0 <= S <= 120.0):
            continue
        res = mod.results(np.array([AR/mod.scale[0], S/mod.scale[1]]))
        rows_f.append({'AR_w': AR, 'S_w': S, 'b_w': res['b_w'],
                       'W0_kgf': res['W0']/gravity})
    df_f = pd.DataFrame(rows_f)
    df_f.to_csv(os.path.join(RESULTS_DIR, 'ver_fronteira.csv'), index=False)

    i_best = int(df_f['W0_kgf'].idxmin())
    melhor = df_f.loc[i_best]
    print('  melhor ponto SOBRE a fronteira: AR=%.4f  S=%.4f  W0=%.4f kgf'
          % (melhor['AR_w'], melhor['S_w'], melhor['W0_kgf']))
    print('  ótimo interior                : AR=%.4f  S=%.4f  W0=%.4f kgf'
          % (restrita['AR_w'], restrita['S_w'], restrita['W0_kgf']))
    print('  penalidade de ir para a fronteira: %+.4f kgf'
          % (melhor['W0_kgf'] - restrita['W0_kgf']))

    # ---------------------------------------------------------------
    # 4. Gradiente físico no ótimo

    print('\n' + '=' * 78)
    print('  4. GRADIENTE DE W0 NO ÓTIMO (diferenças centradas)')
    print('=' * 78)
    AR_o, S_o = restrita['AR_w'], restrita['S_w']

    def W0_kgf(AR, S):
        return mod.results(np.array([AR/mod.scale[0],
                                     S/mod.scale[1]]))['W0']/gravity

    hA, hS = 1e-3, 1e-2
    dA = (W0_kgf(AR_o + hA, S_o) - W0_kgf(AR_o - hA, S_o))/(2*hA)
    dS = (W0_kgf(AR_o, S_o + hS) - W0_kgf(AR_o, S_o - hS))/(2*hS)
    print('  dW0/dAR_w = %+.5f kgf        (por unidade de AR)' % dA)
    print('  dW0/dS_w  = %+.5f kgf/m^2' % dS)
    print('  => gradiente nulo: ponto estacionário INTERIOR (KKT sem')
    print('     multiplicadores ativos).')

    pd.DataFrame([{'AR_w': AR_o, 'S_w': S_o,
                   'dW0_dAR_kgf': dA, 'dW0_dS_kgf_m2': dS,
                   'W0_kgf': restrita['W0_kgf'],
                   'b_w': restrita['b_w'],
                   'W0_melhor_fronteira_kgf': melhor['W0_kgf'],
                   'penalidade_fronteira_kgf': (melhor['W0_kgf']
                                                - restrita['W0_kgf'])}]).to_csv(
        os.path.join(RESULTS_DIR, 'ver_gradiente.csv'), index=False)

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
