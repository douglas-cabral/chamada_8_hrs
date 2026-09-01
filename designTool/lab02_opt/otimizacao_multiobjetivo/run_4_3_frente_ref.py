'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 4 - Grupo NJ-0502

Etapa 3 - Frente de Pareto de referência pelo método da e-restrição.

O MOGA é um método de ordem zero: em 20 000 avaliações ele chega perto da
frente, mas não a resolve com a precisão do SLSQP. Para ter um padrão de
comparação - e para responder à pergunta 3 do roteiro com números, e não
só com um gráfico - calcula-se aqui a frente "exata" pelo método da
e-restrição:

    min  W0(x)
    s.a. g(x) >= 0
         Wf(x) <= eps

varrendo eps entre as duas âncoras. Cada ponto é um SLSQP completo, com os
mesmos gradientes por diferenças finitas centradas do Problema 3. Como o
problema é bem comportado entre as âncoras, a curva resultante é a frente
de Pareto de referência.

Uso:  python run_4_3_frente_ref.py   (depois de run_4_1_ancoras.py)
'''

# IMPORTS
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from opt_multi import DV_NAMES, gravity
import opt_common as oc
from run_4_1_ancoras import CATEGORIAS, AnchorModel

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_multiobjetivo')

OPTIONS = {'maxiter': 300, 'ftol': 1e-10, 'disp': False}

N_PONTOS = 21     # pontos da varredura de eps

# =========================================


class EpsModel(AnchorModel):
    '''
    Minimiza W0 com a restrição adicional Wf <= eps.

    A restrição extra entra normalizada, como todas as outras:
    (1 - Wf/eps) >= 0.
    '''

    def __init__(self, eps_N, **kwargs):
        self.eps_N = float(eps_N)
        super(EpsModel, self).__init__(DV_NAMES, obj_key='W0', **kwargs)

    def confun_eps(self, x):
        res = self.results(x)
        g = oc.constraint_vector(res, self.con_names)
        return np.append(g, 1.0 - res['W_fuel']/self.eps_N)


def frente_eps(caso, rotulo, b_max, w_max, ancoras):
    '''
    Varre eps entre as duas âncoras da categoria e devolve a frente.
    '''
    oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_max, w_max

    d = ancoras[ancoras['caso'] == caso].set_index('ancora')
    wf_lo = d.loc['min Wf', 'Wf_kgf']      # menor Wf alcançável
    wf_hi = d.loc['min W0', 'Wf_kgf']      # Wf no ponto de menor W0

    print('=' * 78)
    print('  FRENTE DE REFERÊNCIA - %s' % rotulo)
    print('  eps varre Wf de %.2f a %.2f kgf em %d pontos'
          % (wf_lo, wf_hi, N_PONTOS))
    print('=' * 78)
    print('  %10s  %13s  %13s  %8s  %s'
          % ('eps [kgf]', 'W0 [kgf]', 'Wf [kgf]', 'b_w [m]', 'status'))

    rows = []
    for frac in np.linspace(0.0, 1.0, N_PONTOS):
        eps_kgf = wf_lo + frac*(wf_hi - wf_lo)
        model = EpsModel(eps_kgf*gravity)
        cons = [{'type': 'ineq', 'fun': model.confun_eps}]

        t0 = time.time()
        result = minimize(model.objfun, model.x0, constraints=cons,
                          bounds=model.bounds, method='slsqp',
                          options=OPTIONS)
        elapsed = time.time() - t0

        res = model.results(result.x)
        g = oc.constraint_vector(res)
        x_phys = model.to_physical(result.x)

        print('  %10.2f  %13.3f  %13.3f  %8.3f  %s'
              % (eps_kgf, res['W0']/gravity, res['W_fuel']/gravity,
                 res['b_w'], 'ok' if result.success else 'parcial'))

        row = {'caso': caso,
               'rotulo': rotulo,
               'eps_kgf': eps_kgf,
               'W0_kgf': res['W0']/gravity,
               'Wf_kgf': res['W_fuel']/gravity,
               'b_w': res['b_w'],
               'g_min': g.min(),
               'viavel': bool(g.min() >= -1e-4),
               'n_objfun': model.n_objfun,
               'tempo_s': elapsed,
               'sucesso': bool(result.success)}
        for name, value in zip(DV_NAMES, x_phys):
            row[name] = value
        rows.append(row)

    print()
    return rows


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    ancoras = pd.read_csv(os.path.join(RESULTS_DIR,
                                       'moga_ancoras_todas.csv'))

    b_ori, w_ori = oc.B_W_MAX, oc.WHEEL_SPAN_MAX
    rows = []
    try:
        for caso, rotulo, b_max, w_max in CATEGORIAS:
            rows += frente_eps(caso, rotulo, b_max, w_max, ancoras)
    finally:
        oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_ori, w_ori

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'ref_frente_eps.csv'), index=False)

    print('=' * 78)
    print('  AMPLITUDE DA FRENTE DE REFERÊNCIA')
    print('=' * 78)
    for caso, rotulo, _, _ in CATEGORIAS:
        d = df[(df['caso'] == caso) & df['viavel']]
        if d.empty:
            continue
        print('  %-24s  W0: %.2f kgf | Wf: %.2f kgf'
              % (rotulo,
                 d['W0_kgf'].max() - d['W0_kgf'].min(),
                 d['Wf_kgf'].max() - d['Wf_kgf'].min()))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
