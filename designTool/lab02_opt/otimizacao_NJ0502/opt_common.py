'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 3 (Team Aircraft Optimization) - Grupo NJ-0502

Módulo comum aos scripts de otimização da aeronave NJ-0502.

Concentra:
  - a definição das variáveis de projeto e de seus limites;
  - a definição das restrições na forma normalizada g(x) >= 0;
  - o acoplamento com o designTool (função `analyze`);
  - a classe `Model`, que entrega função objetivo, restrições e gradientes
    já normalizados para o `scipy.optimize.minimize`.
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

BASELINE_NAME = 'my_airplane'

# O laço de convergência de W0 dentro do designTool para quando o resíduo
# cai abaixo de 10 N, o que deixa um ruído relativo de ~3e-6 em W0. Esse
# ruído contamina as diferenças finitas, então reaplicamos o ponto fixo
# algumas vezes reinjetando o W0 convergido como novo palpite.
N_REFINE = 4

# Passo das diferenças finitas centradas, em variáveis normalizadas.
H_FD = 1e-4

# =========================================
# VARIÁVEIS DE PROJETO
#
# Os limites são dados em unidades físicas; a normalização é feita pela
# classe Model. A ordem desta lista define a ordem do vetor de projeto.

DESIGN_VARS = [
    ('S_w',     r'$S_w$',        'm$^2$', 1.0,      320.0,      470.0),
    ('AR_w',    r'$AR_w$',       '-',     1.0,        7.5,       12.0),
    ('sweep_w', r'$\Lambda_w$',  'deg',   rad2deg,  25.0*deg2rad, 40.0*deg2rad),
    ('xr_w',    r'$x_{r,w}$',    'm',     1.0,       17.0,       23.0),
    # Piso de controle: valor de PRJ-22. Margem estática é ajustada
    # com xr_w (não com Cht), para não entregar EH pequena ao S&C.
    ('Cht',     r'$C_{ht}$',     '-',     1.0,        0.70,       1.10),
    ('Cvt',     r'$C_{vt}$',     '-',     1.0,        0.050,      0.120),
    ('x_mlg',   r'$x_{mlg}$',    'm',     1.0,       29.0,       36.0),
    # Bitola: fora da fuselagem (D_f/2 = 2,98 m) e inboard do motor
    # (y_n - D_n/2 = 9,03 m). O teto efetivo é o da letra E (13,9 m).
    ('y_mlg',   r'$y_{mlg}$',    'm',     1.0,        4.0,        6.95),
    ('z_lg',    r'$z_{lg}$',     'm',     1.0,       -7.0,       -4.5),
]

DV_INDEX = {spec[0]: i for i, spec in enumerate(DESIGN_VARS)}
DV_NAMES = [spec[0] for spec in DESIGN_VARS]

# =========================================
# RESTRIÇÕES
#
# Todas escritas na forma normalizada g(x) >= 0, de modo que g seja
# adimensional e da ordem da unidade. Assim uma única tolerância do
# otimizador atende a todas elas.

# Limites SUPERIORES da categoria de aeródromo. A letra/grupo é definida
# pela dimensão MAIS exigente (OACI Anexo 14 Sec. 1.3; FAA AC 150/5300-13A).
# Não se impõe o piso da faixa (52 m, 9 m, 18,5 m): um 4E pode ter bitola
# < 9 m ou altura de cauda < 18,5 m e continua 4E, porque a envergadura
# é o elemento mais exigente.
#
# A letra E e o ADG V são "até, mas não incluindo, 65 m". O teto 64,9 m
# deixa folga numérica para não escorregar para 4F / ADG VI.
B_W_MAX = 64.9         # envergadura máxima para permanecer OACI E / FAA V [m]
WHEEL_SPAN_MAX = 13.9  # teto da bitola para não subir a letra F [m]
H_TAIL_MAX = 20.0      # teto da altura para não subir ao ADG VI [m]

# Estação do trem principal na asa: xi = (x_mlg - x_LE)/c_local.
# xi <= 1 garante que o trem ainda esteja sob a planta (recolhimento
# no caixão / carenagem de bordo de fuga). xi >= 0,50 coloca a viga
# na metade traseira, onde fica a longarina traseira.
XI_MLG_MIN = 0.50
XI_MLG_MAX = 1.00

# Limite de sustentacao da empenagem vertical na condicao OEI.
CLV_MAX = 0.75

CONSTRAINTS = [
    ('landing',    r'$\Delta S_{wlan} \geq 0$',
     r'$\Delta S_{wlan}/S_w$',
     lambda r: r['deltaS_wlan']/r['S_w'], 'roteiro'),
    ('SM_fwd',     r'$SM_{fwd} \leq 0{,}25$',
     r'$1 - SM_{fwd}/0{,}25$',
     lambda r: 1.0 - r['SM_fwd']/0.25, 'adicionada'),
    ('SM_aft',     r'$SM_{aft} \geq 0{,}05$',
     r'$SM_{aft}/0{,}05 - 1$',
     lambda r: r['SM_aft']/0.05 - 1.0, 'roteiro'),
    ('SM_aft_max', r'$SM_{aft} \leq 0{,}10$',
     r'$1 - SM_{aft}/0{,}10$',
     lambda r: 1.0 - r['SM_aft']/0.10, 'adicionada'),
    ('nlg_fwd',    r'$f_{nlg,fwd} \leq 0{,}18$',
     r'$1 - f_{nlg,fwd}/0{,}18$',
     lambda r: 1.0 - r['frac_nlg_fwd']/0.18, 'roteiro'),
    ('nlg_aft',    r'$f_{nlg,aft} \geq 0{,}03$',
     r'$f_{nlg,aft}/0{,}03 - 1$',
     lambda r: r['frac_nlg_aft']/0.03 - 1.0, 'roteiro'),
    ('tipback',    r'$\alpha_{tip} \geq 15^\circ$',
     r'$\alpha_{tip}/15^\circ - 1$',
     lambda r: r['alpha_tipback']/(15.0*deg2rad) - 1.0, 'roteiro'),
    ('tailstrike', r'$\alpha_{tail} \geq 10^\circ$',
     r'$\alpha_{tail}/10^\circ - 1$',
     lambda r: r['alpha_tailstrike']/(10.0*deg2rad) - 1.0, 'roteiro'),
    ('overturn',   r'$\phi_{ovt} \leq 63^\circ$',
     r'$1 - \phi_{ovt}/63^\circ$',
     lambda r: 1.0 - r['phi_overturn']/(63.0*deg2rad), 'roteiro'),
    ('tank',       r'$tank\_excess \geq 0$',
     r'$W_{maxfuel}/W_f - 1$',
     lambda r: r['tank_excess'], 'roteiro'),
    ('span',       r'$b_w < 65$ m (teto OACI E)',
     r'$1 - b_w/64{,}9$',
     lambda r: 1.0 - r['b_w']/B_W_MAX, 'roteiro'),
    ('wheelspan',  r'$b_{mlg} < 14$ m (teto OACI E)',
     r'$1 - b_{mlg}/13{,}9$',
     lambda r: 1.0 - r['wheel_span']/WHEEL_SPAN_MAX, 'roteiro'),
    ('height',     r'$h_{tail} < 20$ m (teto FAA V)',
     r'$1 - h_{tail}/20$',
     lambda r: 1.0 - r['h_tail']/H_TAIL_MAX, 'roteiro'),
    ('thrust',     r'$T_0 \geq T_{0,req}$',
     r'$T_0/T_{0,req} - 1$',
     lambda r: r['T0']/r['T0req'] - 1.0, 'adicionada'),
    ('CLv',        r'$C_{Lv} \leq 0{,}75$',
     r'$1 - C_{Lv}/0{,}75$',
     lambda r: 1.0 - r['CLv']/CLV_MAX, 'adicionada'),
    ('gear_te',    r'$\xi_{mlg} \leq 1$ (trem sob a asa)',
     r'$1 - \xi_{mlg}$',
     lambda r: 1.0 - r['xi_mlg']/XI_MLG_MAX, 'adicionada'),
    ('gear_spar',  r'$\xi_{mlg} \geq 0{,}50$ (longarina traseira)',
     r'$\xi_{mlg}/0{,}50 - 1$',
     lambda r: r['xi_mlg']/XI_MLG_MIN - 1.0, 'adicionada'),
]

CON_NAMES = [c[0] for c in CONSTRAINTS]
CON_INDEX = {c[0]: i for i, c in enumerate(CONSTRAINTS)}

# =========================================
# ACOPLAMENTO COM O designTool


def get_baseline():
    '''
    Devolve o dicionário de entradas da aeronave do grupo (NJ-0502).
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
    Reúne num único dicionário plano todas as grandezas usadas pela função
    objetivo e pelas restrições.
    '''
    inp = airplane['inputs']
    geo = airplane['geometry']
    bal = airplane['balance']
    tmt = airplane['thrust_matching']
    lgr = airplane['landing_gear']

    eta = inp['y_mlg']/geo['yt_w']
    x_le_mlg = inp['xr_w'] + eta*(geo['xt_w'] - inp['xr_w'])
    c_mlg = geo['cr_w'] + eta*(geo['ct_w'] - geo['cr_w'])
    xi_mlg = (inp['x_mlg'] - x_le_mlg)/c_mlg

    return {
        'W0': tmt['W0'],
        'W_empty': tmt['W_empty'],
        'W_fuel': tmt['W_fuel'],
        'T0': tmt['T0'],
        'T0req': max(tmt['T0req'].values()),
        'deltaS_wlan': tmt['deltaS_wlan'],
        'S_w': inp['S_w'],
        'SM_fwd': bal['SM_fwd'],
        'SM_aft': bal['SM_aft'],
        'CLv': bal['CLv'],
        'tank_excess': bal['tank_excess'],
        'frac_nlg_fwd': lgr['frac_nlg_fwd'],
        'frac_nlg_aft': lgr['frac_nlg_aft'],
        'alpha_tipback': lgr['alpha_tipback'],
        'alpha_tailstrike': lgr['alpha_tailstrike'],
        'phi_overturn': lgr['phi_overturn'],
        'b_w': geo['b_w'],
        'S_h': geo['S_h'],
        'S_v': geo['S_v'],
        'wheel_span': 2.0*inp['y_mlg'],
        'h_tail': geo['zt_v'] - inp['z_lg'],
        'xi_mlg': xi_mlg,
        'x_le_mlg': x_le_mlg,
        'x_te_mlg': x_le_mlg + c_mlg,
    }


def constraint_vector(res, con_names=None):
    '''
    Avalia as restrições normalizadas a partir do dicionário de saídas.

    `con_names` permite avaliar apenas um subconjunto, o que é usado nos
    estudos de remoção de restrições.
    '''
    if con_names is None:
        specs = CONSTRAINTS
    else:
        lookup = {spec[0]: spec for spec in CONSTRAINTS}
        specs = [lookup[name] for name in con_names]

    return np.array([spec[3](res) for spec in specs])


# =========================================
# MODELO DE OTIMIZAÇÃO


class Model(object):
    '''
    Empacota o designTool como um problema de otimização normalizado.

    As variáveis de projeto ativas são passadas por nome. Todas as demais
    entradas ficam congeladas no valor da aeronave de referência.

    A normalização usa o módulo do valor de referência como escala, de forma
    que o ponto de partida tenha componentes de módulo unitário e que o sinal
    da variável seja preservado (importante para z_lg, que é negativo).
    '''

    def __init__(self, dv_names, baseline_inputs=None, h_fd=H_FD,
                 con_names=None, bounds_phys=None):

        self.dv_names = list(dv_names)
        self.base_inputs = (get_baseline() if baseline_inputs is None
                            else copy.deepcopy(baseline_inputs))
        self.h_fd = h_fd

        # Restrições impostas ao otimizador. O histórico sempre guarda o
        # conjunto completo, para que os estudos de remoção mostrem o que
        # acontece com a restrição desligada.
        self.con_names = list(CON_NAMES if con_names is None else con_names)

        specs = {spec[0]: spec for spec in DESIGN_VARS}
        self.specs = [specs[name] for name in self.dv_names]

        self.x0_phys = np.array([self.base_inputs[name]
                                 for name in self.dv_names])
        self.scale = np.abs(self.x0_phys)

        # Limites em unidades físicas, com possibilidade de sobrescrever
        # os valores padrão para estudar o efeito das caixas de projeto.
        overrides = bounds_phys or {}
        self.limits = [overrides.get(spec[0], (spec[4], spec[5]))
                       for spec in self.specs]

        self.x0 = self.x0_phys/self.scale
        self.bounds = [(lim[0]/s, lim[1]/s)
                       for lim, s in zip(self.limits, self.scale)]

        # Contadores e histórico
        # n_designTool conta avaliações de run_designTool, NÃO iterações
        # do SLSQP. Com diferenças centradas, cada pedido de f custa
        # 1 + 2n avaliações de modelo; os 2n do gradiente não entram em n_f.
        self.n_designTool = 0   # chamadas ao modelo (inclui gradientes)
        self.n_objfun = 0       # avaliações da função objetivo
        self.hist_x = []
        self.hist_f = []
        self.hist_g = []

        self._cache = {}
        self._fd_cache = {}

        # Referência da função objetivo: MTOW da configuração de partida.
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
        Devolve as saídas do designTool para o ponto x, com cache para que
        objetivo, restrições e gradientes compartilhem a mesma análise.
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
        '''Vetor de restrições normalizadas ativas, na forma g(x) >= 0.'''
        return constraint_vector(self.results(x), self.con_names)

    def _finite_differences(self, x):
        '''
        Diferenças finitas centradas do objetivo e das restrições.

        O modelo do designTool tem um laço de ponto fixo interno, então não há
        gradiente analítico disponível. O passo padrão do SLSQP (~1e-8) cairia
        dentro do resíduo de convergência; por isso usamos um passo maior e
        diferenças centradas.
        '''
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

            gradf[i] = (res_p['W0'] - res_m['W0'])/self.W0_ref/(2*self.h_fd)
            jacg[:, i] = (constraint_vector(res_p, self.con_names)
                          - constraint_vector(res_m, self.con_names))/(2*self.h_fd)

        self._fd_cache[key] = (gradf, jacg)
        return gradf, jacg

    def objgrad(self, x):
        return self._finite_differences(x)[0]

    def conjac(self, x):
        return self._finite_differences(x)[1]

    # -------------------------------------

    def summary(self, x):
        '''
        Devolve um dicionário com variáveis físicas, objetivo e restrições
        para um dado ponto de projeto.
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
    Formata as grandezas dimensionais de interesse para impressão no terminal.
    '''
    return [
        ('W0 [kgf]',            res['W0']/gravity),
        ('W_empty [kgf]',       res['W_empty']/gravity),
        ('W_fuel [kgf]',        res['W_fuel']/gravity),
        ('T0 [kgf]',            res['T0']/gravity),
        ('T0req [kgf]',         res['T0req']/gravity),
        ('S_w [m2]',            res['S_w']),
        ('b_w [m]',             res['b_w']),
        ('S_h [m2]',            res['S_h']),
        ('S_v [m2]',            res['S_v']),
        ('deltaS_wlan [m2]',    res['deltaS_wlan']),
        ('SM_fwd [-]',          res['SM_fwd']),
        ('SM_aft [-]',          res['SM_aft']),
        ('CLv [-]',             res['CLv']),
        ('tank_excess [-]',     res['tank_excess']),
        ('frac_nlg_fwd [-]',    res['frac_nlg_fwd']),
        ('frac_nlg_aft [-]',    res['frac_nlg_aft']),
        ('alpha_tipback [deg]', res['alpha_tipback']*rad2deg),
        ('alpha_tail [deg]',    res['alpha_tailstrike']*rad2deg),
        ('phi_overturn [deg]',  res['phi_overturn']*rad2deg),
        ('h_tail [m]',          res['h_tail']),
        ('wheel_span [m]',      res['wheel_span']),
        ('xi_mlg [-]',          res['xi_mlg']),
    ]
