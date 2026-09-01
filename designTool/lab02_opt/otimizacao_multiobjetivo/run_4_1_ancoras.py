'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 4 - Grupo NJ-0502

Etapa 1 - Âncoras da frente de Pareto.

A pergunta 3 do Problema 4 pede como usar os resultados do Problema 3 para
verificar a convergência da otimização multiobjetivo. A resposta operacional
é: o ótimo do Problema 3 minimiza W0 sujeito às MESMAS restrições, logo ele
é, por definição, o extremo da frente de Pareto na direção de W0. Este
script calcula os dois extremos por SLSQP:

  - âncora A: min W0  (reproduz o Problema 3);
  - âncora B: min Wf  (o outro extremo da frente).

Os dois pontos delimitam a caixa onde a frente do MOGA precisa cair.

Uso:  python run_4_1_ancoras.py
'''

# IMPORTS
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# opt_multi precisa vir primeiro: é ele que põe a pasta do Problema 3
# no sys.path, tornando `opt_common` importável a partir daqui.
from opt_multi import CONSTRAINTS, DV_NAMES, gravity
import opt_common as oc

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_multiobjetivo')

OPTIONS = {'maxiter': 300, 'ftol': 1e-8, 'disp': False}

TOL_ATIVA = 1e-4

# As duas categorias de aeródromo estudadas. A primeira é a do roteiro;
# a segunda serve para identificar QUEM trunca a frente de Pareto.
CATEGORIAS = [
    ('base',   'letra E (roteiro)',       64.9, 13.9),
    ('letraF', 'letra F (teto relaxado)', 79.9, 15.9),
]

# =========================================


class AnchorModel(oc.Model):
    '''
    Model do Problema 3 com a função objetivo trocável.

    `obj_key` escolhe a grandeza minimizada ('W0' ou 'W_fuel'). Todo o
    resto - variáveis, limites, restrições, normalização e diferenças
    finitas centradas - é herdado sem alteração, de modo que as duas
    âncoras resolvam exatamente o mesmo problema restringido.
    '''

    def __init__(self, dv_names, obj_key='W0', **kwargs):
        self.obj_key = obj_key
        super(AnchorModel, self).__init__(dv_names, **kwargs)
        self.obj_ref = self.res0[obj_key]

    def objfun(self, x):
        res = self.results(x)
        f = res[self.obj_key]/self.obj_ref

        self.n_objfun += 1
        self.hist_x.append(np.asarray(x, dtype=float).copy())
        self.hist_f.append(f)
        self.hist_g.append(oc.constraint_vector(res))

        return f

    def _finite_differences(self, x):
        key = self._key(x)
        if key in self._fd_cache:
            return self._fd_cache[key]

        x = np.asarray(x, dtype=float)
        n = len(x)
        gradf = np.zeros(n)
        jacg = np.zeros((len(self.con_names), n))

        for i in range(n):
            step = np.zeros(n)
            step[i] = self.h_fd

            res_p = self.results(x + step)
            res_m = self.results(x - step)

            gradf[i] = ((res_p[self.obj_key] - res_m[self.obj_key])
                        / self.obj_ref/(2*self.h_fd))
            jacg[:, i] = (oc.constraint_vector(res_p, self.con_names)
                          - oc.constraint_vector(res_m, self.con_names)) \
                / (2*self.h_fd)

        self._fd_cache[key] = (gradf, jacg)
        return gradf, jacg


def solve(obj_key):
    model = AnchorModel(DV_NAMES, obj_key=obj_key)
    cons = [{'type': 'ineq', 'fun': model.confun, 'jac': model.conjac}]

    t0 = time.time()
    result = minimize(model.objfun, model.x0, jac=model.objgrad,
                      constraints=cons, bounds=model.bounds,
                      method='slsqp', options=OPTIONS)
    elapsed = time.time() - t0

    res = model.results(result.x)
    g = oc.constraint_vector(res)
    return model, result, elapsed, res, g


# =========================================

def roda_categoria(caso, rotulo, b_max, w_max):
    """
    Calcula as duas âncoras para uma categoria de aeródromo.
    """
    oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_max, w_max

    print('#' * 78)
    print('#  CATEGORIA "%s": %s  (b_w < %.1f m, b_mlg < %.1f m)'
          % (caso, rotulo, b_max, w_max))
    print('#' * 78)

    rows = []
    for obj_key, nome in [('W0', 'min W0'), ('W_fuel', 'min Wf')]:

        print('=' * 78)
        print('  ÂNCORA: %s' % nome)
        print('=' * 78)

        model, result, elapsed, res, g = solve(obj_key)
        x_phys = model.to_physical(result.x)

        print('  status          : %s' % result.message)
        print('  iterações SLSQP : %d' % result.nit)
        print('  chamadas de f   : %d' % model.n_objfun)
        print('  aval. designTool: %d' % model.n_designTool)
        print('  tempo           : %.2f s' % elapsed)
        print('  W0 = %11.2f kgf     Wf = %11.2f kgf'
              % (res['W0']/gravity, res['W_fuel']/gravity))
        print('  b_w = %.3f m' % res['b_w'])
        print('  menor g         : %+.6f  (%s)'
              % (g.min(), 'viável' if g.min() >= -TOL_ATIVA else 'INVIÁVEL'))
        ativas = [spec[0] for spec, gi in zip(CONSTRAINTS, g)
                  if abs(gi) <= TOL_ATIVA]
        print('  ativas          : %s' % (', '.join(ativas) or 'nenhuma'))
        print('  variáveis       :')
        for name, value in zip(DV_NAMES, x_phys):
            print('     %-8s %12.5f' % (name, value))
        print()

        row = {'caso': caso,
               'rotulo': rotulo,
               'ancora': nome,
               'obj': obj_key,
               'b_w': res['b_w'],
               'W0_kgf': res['W0']/gravity,
               'Wf_kgf': res['W_fuel']/gravity,
               'f1': res['W0']/model.res0['W0'],
               'f2': res['W_fuel']/model.res0['W_fuel'],
               'n_objfun': model.n_objfun,
               'n_designTool': model.n_designTool,
               'tempo_s': elapsed,
               'g_min': g.min(),
               'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
               'ativas': '; '.join(ativas),
               'sucesso': bool(result.success)}
        for name, value in zip(DV_NAMES, x_phys):
            row[name] = value
        rows.append(row)

    return rows


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Guarda os tetos originais para restaurar ao fim.
    b_ori, w_ori = oc.B_W_MAX, oc.WHEEL_SPAN_MAX

    rows = []
    try:
        for caso, rotulo, b_max, w_max in CATEGORIAS:
            rows += roda_categoria(caso, rotulo, b_max, w_max)
    finally:
        oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_ori, w_ori

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(RESULTS_DIR, 'moga_ancoras_todas.csv'),
                  index=False)

    # O arquivo `moga_ancoras.csv` guarda o caso do roteiro, indexado
    # pelo nome da âncora, que é como as figuras e tabelas o consomem.
    df = df_all[df_all['caso'] == 'base'].set_index('ancora')
    df.to_csv(os.path.join(RESULTS_DIR, 'moga_ancoras.csv'))

    # Referência de PRJ-22 (mesmo ponto de partida de todas as âncoras)
    model_ref = AnchorModel(DV_NAMES, obj_key='W0')
    pd.DataFrame([{
        'W0_kgf': model_ref.res0['W0']/gravity,
        'Wf_kgf': model_ref.res0['W_fuel']/gravity,
    }]).to_csv(os.path.join(RESULTS_DIR, 'moga_referencia.csv'), index=False)

    print('=' * 78)
    print('  CAIXA DA FRENTE DE PARETO (categoria do roteiro)')
    print('=' * 78)
    print(df[['W0_kgf', 'Wf_kgf', 'b_w', 'n_ativas']].to_string(
        float_format=lambda v: '%12.2f' % v))
    print()
    print('  W0 na frente deve ficar em [%.2f ; %.2f] kgf'
          % (df.loc['min W0', 'W0_kgf'], df.loc['min Wf', 'W0_kgf']))
    print('  Wf na frente deve ficar em [%.2f ; %.2f] kgf'
          % (df.loc['min Wf', 'Wf_kgf'], df.loc['min W0', 'Wf_kgf']))

    print()
    print('  Amplitude da frente por categoria:')
    for caso, rotulo, _, _ in CATEGORIAS:
        d = df_all[df_all['caso'] == caso].set_index('ancora')
        dW0 = abs(d.loc['min Wf', 'W0_kgf'] - d.loc['min W0', 'W0_kgf'])
        dWf = abs(d.loc['min W0', 'Wf_kgf'] - d.loc['min Wf', 'Wf_kgf'])
        print('    %-24s  dW0 = %8.2f kgf | dWf = %8.2f kgf'
              % (rotulo, dW0, dWf))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
