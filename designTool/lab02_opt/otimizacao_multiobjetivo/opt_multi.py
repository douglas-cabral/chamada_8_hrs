'''
PRJ-23 Lab 02 - Problema 4. Problema bi-objetivo (W0, Wf) para o pymoo.
Herdado da formulação do Problema 3 (opt_common).
'''

import copy
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3_DIR = os.path.join(os.path.dirname(_HERE), 'otimizacao_NJ0502')
if _P3_DIR not in sys.path:
    sys.path.insert(0, _P3_DIR)

from opt_common import (CONSTRAINTS, CON_NAMES, DESIGN_VARS, DV_INDEX,
                        DV_NAMES, constraint_vector, extract, get_baseline,
                        gravity, physical_report, rad2deg, run_designTool)

from pymoo.core.problem import ElementwiseProblem

OBJ_LABELS = [r'$W_0$ [kgf]', r'$W_f$ [kgf]']
OBJ_KEYS = ['W0', 'W_fuel']
TOL_VIAVEL = 1e-4
PENAL_F = 1.0e3
PENAL_G = -1.0e3
_CHAVES_FINITAS = ('W0', 'W_empty', 'W_fuel', 'T0', 'T0req', 'b_w',
                   'SM_fwd', 'SM_aft', 'CLv', 'tank_excess')


class MultiObjModel(object):
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
        self.n_invalido = 0
        self._cache = {}

        self.res0 = self.results(self.x0)
        self.W0_ref = self.res0['W0']
        self.Wf_ref = self.res0['W_fuel']

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_cache'] = {}
        return state

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

    def objectives(self, x):
        res = self.results(x)
        if not res['_valido']:
            return np.full(2, PENAL_F)
        return np.array([res['W0']/self.W0_ref,
                         res['W_fuel']/self.Wf_ref])

    def constraints_g(self, x):
        res = self.results(x)
        if not res['_valido']:
            return np.full(len(self.con_names), PENAL_G)
        g = constraint_vector(res, self.con_names)
        return np.where(np.isfinite(g), g, PENAL_G)

    def summary(self, x):
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


class NJ0502BiObj(ElementwiseProblem):
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


def frame_from_X(model, X, F=None):
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
