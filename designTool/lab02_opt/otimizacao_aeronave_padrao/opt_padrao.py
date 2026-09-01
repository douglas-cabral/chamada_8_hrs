'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 2 (Default Aircraft Optimization) - Grupo NJ-0502

Módulo comum aos scripts de otimização da aeronave padrão (Fokker 100).

Concentra:
  - a definição das duas variáveis de projeto e de seus limites;
  - a restrição de envergadura na forma normalizada g(x) >= 0;
  - o acoplamento com o designTool (função `analyze`);
  - a classe `Model`, que entrega função objetivo e restrição já
    normalizadas para o `scipy.optimize.minimize`.

Diferença em relação ao Problema 3: aqui o roteiro pede explicitamente que
o otimizador estime as derivadas por diferenças finitas, então a classe
NÃO expõe jacobianos. Todo o resto (normalização, refino do ponto fixo de
W0, contadores) segue a mesma convenção de `otimizacao_NJ0502/opt_common.py`.
'''

# IMPORTS
import copy
import os
import sys

import numpy as np


# O pacote designTool fica na raiz do repositório (dois ou mais níveis
# acima desta pasta). Procuramos analyze.py para que os scripts rodem
# de qualquer profundidade dentro de lab02_opt.
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

# =========================================
# CONSTANTES

deg2rad = np.pi/180.0
rad2deg = 180.0/np.pi

BASELINE_NAME = 'fokker100'

# O laço de convergência de W0 dentro do designTool para quando o resíduo
# cai abaixo de 10 N. Reinjetamos o W0 convergido como novo palpite algumas
# vezes para eliminar esse resíduo residual das diferenças finitas.
# Mesma convenção do Problema 3.
N_REFINE = 4

# Ponto de partida imposto pelo roteiro (Sec. 2).
AR_W_START = 7.5
S_W_START = 90.0

# Teto de envergadura imposto pelo roteiro [m].
B_W_MAX = 30.0

# =========================================
# VARIÁVEIS DE PROJETO
#
# (nome, rótulo LaTeX, unidade, fator p/ unidade de exibição, lim_inf, lim_sup)
# A ordem desta lista define a ordem do vetor de projeto.

DESIGN_VARS = [
    ('AR_w', r'$AR_w$', '-',       1.0,  7.0,  12.0),
    ('S_w',  r'$S_w$',  'm$^2$',   1.0, 80.0, 120.0),
]

DV_NAMES = [spec[0] for spec in DESIGN_VARS]
DV_INDEX = {spec[0]: i for i, spec in enumerate(DESIGN_VARS)}

# =========================================
# RESTRIÇÕES
#
# Forma normalizada g(x) >= 0, adimensional e da ordem da unidade, como
# recomenda o roteiro. Aqui há uma única desigualdade.

CONSTRAINTS = [
    ('span', r'$b_w \leq 30$ m', r'$1 - b_w/30$',
     lambda r: 1.0 - r['b_w']/B_W_MAX, 'roteiro'),
]

CON_NAMES = [c[0] for c in CONSTRAINTS]

# =========================================
# ACOPLAMENTO COM O designTool


def get_baseline():
    '''
    Devolve o dicionário de entradas da aeronave padrão (Fokker 100).
    '''
    return copy.deepcopy(standard_airplane(BASELINE_NAME)['inputs'])


def run_designTool(inputs):
    '''
    Executa o designTool e refina o ponto fixo de W0 reinjetando o valor
    convergido como novo palpite. Devolve o dicionário completo da aeronave.
    '''
    airplane = {'inputs': copy.deepcopy(inputs)}
    analyze(airplane, print_log=False, plot=False)

    for _ in range(N_REFINE):
        airplane['inputs']['W0_guess'] = airplane['thrust_matching']['W0']
        analyze(airplane, print_log=False, plot=False)

    return airplane


def extract(airplane):
    '''
    Reúne num único dicionário plano as grandezas usadas pela função
    objetivo, pela restrição e pelas tabelas do relatório.
    '''
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
    '''
    Avalia as restrições normalizadas a partir do dicionário de saídas.
    '''
    return np.array([spec[3](res) for spec in CONSTRAINTS])


# =========================================
# MODELO DE OTIMIZAÇÃO


class Model(object):
    '''
    Empacota o designTool como um problema de otimização normalizado.

    A normalização usa o valor do ponto de partida como escala, de modo que
    x0 tenha componentes de módulo unitário. Assim uma única tolerância do
    SLSQP serve para o objetivo e para a restrição, e o passo relativo de
    diferenças finitas do SciPy fica bem condicionado nas duas variáveis.

    Ao contrário do Problema 3, esta classe não expõe gradientes: o roteiro
    manda deixar o otimizador estimá-los por diferenças finitas.
    '''

    def __init__(self, x0_phys=None, baseline_inputs=None):

        self.dv_names = list(DV_NAMES)
        self.base_inputs = (get_baseline() if baseline_inputs is None
                            else copy.deepcopy(baseline_inputs))

        # Ponto de partida do roteiro: AR_w = 7,5 e S_w = 90 m2.
        if x0_phys is None:
            x0_phys = np.array([AR_W_START, S_W_START])
        self.x0_phys = np.asarray(x0_phys, dtype=float)
        self.scale = np.abs(self.x0_phys)

        self.limits = [(spec[4], spec[5]) for spec in DESIGN_VARS]
        self.x0 = self.x0_phys/self.scale
        self.bounds = [(lim[0]/s, lim[1]/s)
                       for lim, s in zip(self.limits, self.scale)]

        # Contadores e histórico.
        # n_designTool conta avaliações de run_designTool (inclui as usadas
        # pelo SciPy para montar as diferenças finitas). n_objfun conta
        # apenas os pedidos de f(x) feitos pelo otimizador.
        self.n_designTool = 0
        self.n_objfun = 0
        self.n_confun = 0
        self.hist_x = []
        self.hist_f = []
        self.hist_g = []

        self._cache = {}

        # Referência da função objetivo: MTOW do ponto de partida.
        self.res0 = self.results(self.x0)
        self.W0_ref = self.res0['W0']

    # -------------------------------------

    def to_physical(self, x):
        '''Converte o vetor normalizado para unidades físicas.'''
        return np.asarray(x, dtype=float)*self.scale

    def build_inputs(self, x):
        '''Monta o dicionário de entradas do designTool para um dado x.'''
        inputs = copy.deepcopy(self.base_inputs)
        for name, value in zip(self.dv_names, self.to_physical(x)):
            inputs[name] = value
        return inputs

    def _key(self, x):
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def results(self, x):
        '''
        Saídas do designTool no ponto x, com cache para que objetivo e
        restrição compartilhem a mesma análise.
        '''
        key = self._key(x)
        if key not in self._cache:
            self._cache[key] = extract(run_designTool(self.build_inputs(x)))
            self.n_designTool += 1
        return self._cache[key]

    # -------------------------------------
    # Interface para o scipy

    def objfun(self, x):
        '''Função objetivo normalizada: MTOW dividido pelo MTOW de partida.'''
        res = self.results(x)
        f = res['W0']/self.W0_ref

        self.n_objfun += 1
        self.hist_x.append(np.asarray(x, dtype=float).copy())
        self.hist_f.append(f)
        self.hist_g.append(constraint_vector(res))

        return f

    def confun(self, x):
        '''Restrição normalizada na forma g(x) >= 0.'''
        self.n_confun += 1
        return constraint_vector(self.results(x))

    # -------------------------------------

    def summary(self, x):
        '''
        Dicionário com variáveis físicas, objetivo e restrição num ponto.
        '''
        res = self.results(x)
        out = {'W0_kgf': res['W0']/gravity,
               'f': res['W0']/self.W0_ref}
        for name, value in zip(self.dv_names, self.to_physical(x)):
            out[name] = value
        for spec, g in zip(CONSTRAINTS, constraint_vector(res)):
            out['g_' + spec[0]] = g
        out.update({'raw_' + k: v for k, v in res.items()})
        return out


# =========================================


def physical_report(res):
    '''
    Formata as grandezas dimensionais de interesse para impressão e tabelas.
    '''
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
