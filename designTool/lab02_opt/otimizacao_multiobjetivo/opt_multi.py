'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 4 (Multiobjective Optimization) - Grupo NJ-0502

Módulo comum aos scripts de otimização multiobjetivo da NJ-0502.

O roteiro manda "repetir o problema acima com um objetivo adicional", isto
é: as MESMAS variáveis de projeto e as MESMAS restrições do Problema 3,
agora com dois objetivos, W0 e Wf. Para que a comparação da pergunta 3
(usar a Sec. 3 para verificar a convergência) seja legítima, este módulo
NÃO redefine o problema: importa `opt_common.py` da pasta do Problema 3 e
reaproveita DESIGN_VARS, CONSTRAINTS e o acoplamento com o designTool.
Assim existe uma única fonte de verdade para a formulação.

Convenções:
  - variáveis normalizadas por |x_ref| (idem Problema 3);
  - objetivos normalizados pelos valores da configuração de PRJ-22:
        f1 = W0/W0_ref     f2 = Wf/Wf_ref
  - restrições internas na forma g(x) >= 0; o pymoo usa G(x) <= 0,
    então entrega-se G = -g.
'''

# IMPORTS
import copy
import os
import sys

import numpy as np

# ---------------------------------------------------------------------
# Reaproveita a formulação do Problema 3 (pasta irmã).

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3_DIR = os.path.join(os.path.dirname(_HERE), 'otimizacao_NJ0502')
if _P3_DIR not in sys.path:
    sys.path.insert(0, _P3_DIR)

from opt_common import (CONSTRAINTS, CON_NAMES, DESIGN_VARS, DV_INDEX,
                        DV_NAMES, constraint_vector, extract, get_baseline,
                        gravity, physical_report, rad2deg, run_designTool)

from pymoo.core.problem import ElementwiseProblem

# =========================================
# CONSTANTES

# Nomes dos objetivos, para rótulos de figuras e tabelas.
OBJ_LABELS = [r'$W_0$ [kgf]', r'$W_f$ [kgf]']
OBJ_KEYS = ['W0', 'W_fuel']

# Folga numérica de viabilidade usada na pós-análise.
TOL_VIAVEL = 1e-4

# Penalidade para indivíduos em que o designTool não fecha.
#
# O SLSQP do Problema 3 só visita a vizinhança de um projeto viável, mas o
# MOGA amostra a CAIXA INTEIRA. Em cantos extremos (asa minúscula com
# alcance de 8 000 nm, por exemplo) o laço de ponto fixo de W0 diverge e o
# designTool devolve inf/NaN. Um NaN em F quebra a ordenação por não
# dominância do NSGA-II - comparações com NaN são sempre falsas, e o
# indivíduo passa a "não ser dominado por ninguém". Por isso todo resultado
# não finito é convertido num ponto finito, péssimo e fortemente inviável,
# que o algoritmo descarta pela regra de dominância restringida.
PENAL_F = 1.0e3     # valor dos objetivos normalizados num ponto inválido
PENAL_G = -1.0e3    # valor das restrições g >= 0 num ponto inválido

# Grandezas que precisam ser finitas para o ponto ser considerado válido.
_CHAVES_FINITAS = ('W0', 'W_empty', 'W_fuel', 'T0', 'T0req', 'b_w',
                   'SM_fwd', 'SM_aft', 'CLv', 'tank_excess')


# =========================================
# MODELO


class MultiObjModel(object):
    '''
    Empacota o designTool como problema BI-objetivo normalizado.

    Reaproveita as nove variáveis e as dezessete restrições do Problema 3.
    Ao contrário da classe `Model` daquele problema, não há gradientes:
    o MOGA é um método de ordem zero.
    '''

    def __init__(self, dv_names=None, baseline_inputs=None, con_names=None):

        self.dv_names = list(DV_NAMES if dv_names is None else dv_names)
        self.base_inputs = (get_baseline() if baseline_inputs is None
                            else copy.deepcopy(baseline_inputs))
        self.con_names = list(CON_NAMES if con_names is None else con_names)

        specs = {spec[0]: spec for spec in DESIGN_VARS}
        self.specs = [specs[name] for name in self.dv_names]

        self.x0_phys = np.array([self.base_inputs[name]
                                 for name in self.dv_names])
        self.scale = np.abs(self.x0_phys)

        self.limits = [(spec[4], spec[5]) for spec in self.specs]
        self.x0 = self.x0_phys/self.scale
        self.xl = np.array([lim[0]/s
                            for lim, s in zip(self.limits, self.scale)])
        self.xu = np.array([lim[1]/s
                            for lim, s in zip(self.limits, self.scale)])

        self.n_designTool = 0
        self.n_invalido = 0     # pontos em que o designTool não fechou
        self._cache = {}

        # Referências dos dois objetivos: configuração de PRJ-22.
        self.res0 = self.results(self.x0)
        self.W0_ref = self.res0['W0']
        self.Wf_ref = self.res0['W_fuel']

    # -------------------------------------

    def to_physical(self, x):
        return np.asarray(x, dtype=float)*self.scale

    def build_inputs(self, x):
        inputs = copy.deepcopy(self.base_inputs)
        for name, value in zip(self.dv_names, self.to_physical(x)):
            inputs[name] = value
        return inputs

    def _key(self, x):
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def results(self, x):
        '''
        Saídas do designTool no ponto x, com cache.

        Um ponto em que a análise levanta exceção ou devolve grandezas não
        finitas é marcado com `_valido = False` em vez de propagar NaN.
        '''
        key = self._key(x)
        if key not in self._cache:
            try:
                res = extract(run_designTool(self.build_inputs(x)))
                valido = all(np.isfinite(res[k]) for k in _CHAVES_FINITAS)
            except (ValueError, ArithmeticError, KeyError, TypeError):
                res, valido = {}, False

            if not valido:
                self.n_invalido += 1
            res['_valido'] = valido
            self._cache[key] = res
            self.n_designTool += 1
        return self._cache[key]

    # -------------------------------------

    def objectives(self, x):
        '''Vetor [f1, f2] = [W0/W0_ref, Wf/Wf_ref].'''
        res = self.results(x)
        if not res['_valido']:
            return np.full(2, PENAL_F)
        return np.array([res['W0']/self.W0_ref,
                         res['W_fuel']/self.Wf_ref])

    def constraints_g(self, x):
        '''Restrições na convenção do relatório: g(x) >= 0.'''
        res = self.results(x)
        if not res['_valido']:
            return np.full(len(self.con_names), PENAL_G)
        g = constraint_vector(res, self.con_names)
        # Uma restrição isolada ainda pode sair não finita mesmo com as
        # grandezas principais válidas; trata-se do mesmo modo.
        return np.where(np.isfinite(g), g, PENAL_G)

    def summary(self, x):
        '''Variáveis físicas, objetivos e restrições num ponto de projeto.'''
        res = self.results(x)
        f = self.objectives(x)

        out = {'valido': bool(res['_valido']),
               'f1': f[0],
               'f2': f[1]}
        if res['_valido']:
            out['W0_kgf'] = res['W0']/gravity
            out['Wf_kgf'] = res['W_fuel']/gravity
        else:
            out['W0_kgf'] = np.nan
            out['Wf_kgf'] = np.nan

        for name, value in zip(self.dv_names, self.to_physical(x)):
            out[name] = value

        g_full = (constraint_vector(res) if res['_valido']
                  else np.full(len(CONSTRAINTS), PENAL_G))
        for spec, g in zip(CONSTRAINTS, g_full):
            out['g_' + spec[0]] = g

        out.update({'raw_' + k: v for k, v in res.items()
                    if k != '_valido'})
        return out


# =========================================
# PROBLEMA pymoo


class NJ0502BiObj(ElementwiseProblem):
    '''
    Interface do problema bi-objetivo para o pymoo.

    n_var = 9, n_obj = 2, n_ieq_constr = 17.
    O pymoo minimiza F e exige G <= 0; como o relatório escreve as
    restrições como g >= 0, entrega-se G = -g.
    '''

    def __init__(self, model=None, **kwargs):
        self.model = model if model is not None else MultiObjModel()
        super().__init__(n_var=len(self.model.dv_names),
                         n_obj=2,
                         n_ieq_constr=len(self.model.con_names),
                         xl=self.model.xl,
                         xu=self.model.xu,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.model.objectives(x)
        out['G'] = -self.model.constraints_g(x)


# =========================================


def frame_from_X(model, X, F=None):
    '''
    Monta uma lista de dicionários (pronta para DataFrame) a partir de uma
    população de vetores de projeto normalizados.
    '''
    rows = []
    for i, x in enumerate(np.atleast_2d(X)):
        row = model.summary(x)
        row['idx'] = i
        if F is not None:
            row['f1_moga'] = F[i, 0]
            row['f2_moga'] = F[i, 1]
        g = model.constraints_g(x)
        row['g_min'] = g.min()
        row['viavel'] = bool(g.min() >= -TOL_VIAVEL)
        rows.append(row)
    return rows
