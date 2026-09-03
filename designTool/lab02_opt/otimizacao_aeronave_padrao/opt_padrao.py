'''
PRJ-23 Lab 02 - Problema 2. Modelo da aeronave padrão (Fokker 100).
Duas variáveis, uma restrição; o SLSQP estima os gradientes.
'''

import copy
import os
import sys

import numpy as np


def _find_project_root(start):
    here = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(here, 'designTool', 'analyze.py')):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError(
                'Pacote designTool não encontrado a partir de %s' % start)
        here = parent


_PROJECT_ROOT = _find_project_root(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from designTool.analyze import analyze
from designTool.standard_airplane import standard_airplane
from designTool.constants import gravity

deg2rad = np.pi/180.0
rad2deg = 180.0/np.pi

BASELINE_NAME = 'fokker100'
N_REFINE = 4
AR_W_START = 7.5
S_W_START = 90.0
B_W_MAX = 30.0

# (nome, rótulo LaTeX, unidade, fator de exibição, lim_inf, lim_sup)

DESIGN_VARS = [
    ('AR_w', r'$AR_w$', '-',       1.0,  7.0,  12.0),
    ('S_w',  r'$S_w$',  'm$^2$',   1.0, 80.0, 120.0),
]

DV_NAMES = [spec[0] for spec in DESIGN_VARS]
DV_INDEX = {spec[0]: i for i, spec in enumerate(DESIGN_VARS)}

CONSTRAINTS = [
    ('span', r'$b_w \leq 30$ m', r'$1 - b_w/30$',
     lambda r: 1.0 - r['b_w']/B_W_MAX, 'roteiro'),
]

CON_NAMES = [c[0] for c in CONSTRAINTS]

def get_baseline():
    return copy.deepcopy(standard_airplane(BASELINE_NAME)['inputs'])


def run_designTool(inputs):
    airplane = {'inputs': copy.deepcopy(inputs)}
    analyze(airplane, print_log=False, plot=False)

    for _ in range(N_REFINE):
        airplane['inputs']['W0_guess'] = airplane['thrust_matching']['W0']
        analyze(airplane, print_log=False, plot=False)

    return airplane


def extract(airplane):
    inp = airplane['inputs']
    geo = airplane['geometry']
    tmt = airplane['thrust_matching']

    return {
        'W0': tmt['W0'],
        'W_empty': tmt['W_empty'],
        'W_fuel': tmt['W_fuel'],
        'T0': tmt['T0'],
        'T0req': max(tmt['T0req'].values()),
        'deltaS_wlan': tmt['deltaS_wlan'],
        'S_w': inp['S_w'],
        'AR_w': inp['AR_w'],
        'b_w': geo['b_w'],
        'cr_w': geo['cr_w'],
        'ct_w': geo['ct_w'],
        'cm_w': geo['cm_w'],
        'S_h': geo['S_h'],
        'S_v': geo['S_v'],
    }


def constraint_vector(res):
    return np.array([spec[3](res) for spec in CONSTRAINTS])


class Model(object):
    def __init__(self, x0_phys=None, baseline_inputs=None):

        self.dv_names = list(DV_NAMES)
        self.base_inputs = (get_baseline() if baseline_inputs is None
                            else copy.deepcopy(baseline_inputs))

        if x0_phys is None:
            x0_phys = np.array([AR_W_START, S_W_START])
        self.x0_phys = np.asarray(x0_phys, dtype=float)
        self.scale = np.abs(self.x0_phys)

        self.limits = [(spec[4], spec[5]) for spec in DESIGN_VARS]
        self.x0 = self.x0_phys/self.scale
        self.bounds = [(lim[0]/s, lim[1]/s)
                       for lim, s in zip(self.limits, self.scale)]

        self.n_designTool = 0
        self.n_objfun = 0
        self.n_confun = 0
        self.hist_x = []
        self.hist_f = []
        self.hist_g = []

        self._cache = {}
        self.res0 = self.results(self.x0)
        self.W0_ref = self.res0['W0']

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
            self._cache[key] = extract(run_designTool(self.build_inputs(x)))
            self.n_designTool += 1
        return self._cache[key]

    def objfun(self, x):
        res = self.results(x)
        f = res['W0']/self.W0_ref

        self.n_objfun += 1
        self.hist_x.append(np.asarray(x, dtype=float).copy())
        self.hist_f.append(f)
        self.hist_g.append(constraint_vector(res))

        return f

    def confun(self, x):
        self.n_confun += 1
        return constraint_vector(self.results(x))

    def summary(self, x):
        res = self.results(x)
        out = {'W0_kgf': res['W0']/gravity,
               'f': res['W0']/self.W0_ref}
        for name, value in zip(self.dv_names, self.to_physical(x)):
            out[name] = value
        for spec, g in zip(CONSTRAINTS, constraint_vector(res)):
            out['g_' + spec[0]] = g
        out.update({'raw_' + k: v for k, v in res.items()})
        return out


def physical_report(res):
    return [
        ('W0 [kgf]',         res['W0']/gravity),
        ('W_empty [kgf]',    res['W_empty']/gravity),
        ('W_fuel [kgf]',     res['W_fuel']/gravity),
        ('T0 [kgf]',         res['T0']/gravity),
        ('T0req [kgf]',      res['T0req']/gravity),
        ('AR_w [-]',         res['AR_w']),
        ('S_w [m2]',         res['S_w']),
        ('b_w [m]',          res['b_w']),
        ('cr_w [m]',         res['cr_w']),
        ('ct_w [m]',         res['ct_w']),
        ('S_h [m2]',         res['S_h']),
        ('S_v [m2]',         res['S_v']),
        ('deltaS_wlan [m2]', res['deltaS_wlan']),
    ]
