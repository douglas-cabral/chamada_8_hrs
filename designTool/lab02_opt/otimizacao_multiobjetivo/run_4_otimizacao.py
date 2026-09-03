'''
PRJ-23 Lab 02 - Problema 4. Âncoras SLSQP, NSGA-II, frente de referência e figuras.
Uso: python run_4_otimizacao.py
'''

import os
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
from scipy.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from opt_multi import (CONSTRAINTS, DV_NAMES, MultiObjModel, NJ0502BiObj,
                       TOL_VIAVEL, frame_from_X, gravity)
import opt_common as oc
from opt_common import run_designTool

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_multiobjetivo')

OPTIONS_ANC = {'maxiter': 300, 'ftol': 1e-8, 'disp': False}
OPTIONS_EPS = {'maxiter': 300, 'ftol': 1e-10, 'disp': False}
TOL_ATIVA = 1e-4
N_PONTOS = 21
POP_SIZE, N_GEN, SEED = 100, 400, 42
N_SEMEADOS, DISPERSAO_SEMENTE = 10, 0.02
ETA_CROSSOVER, PROB_CROSSOVER, ETA_MUTATION = 15, 0.9, 20

CATEGORIAS = [
    ('base', 'letra E (roteiro)', 64.9, 13.9),
    ('letraF', 'letra F (teto relaxado)', 79.9, 15.9),
]
CASOS = {
    'base': {'rotulo': 'letra E (roteiro)', 'B_W_MAX': 64.9,
             'WHEEL_SPAN_MAX': 13.9},
    'letraF': {'rotulo': 'letra F (teto relaxado)', 'B_W_MAX': 79.9,
               'WHEEL_SPAN_MAX': 15.9},
}

COR_MOGA, COR_REF = '#1f4e79', '#c53030'
COR_ANC_W0, COR_ANC_WF, COR_PRJ22 = '#c53030', '#2f855a', '#6b46c1'
CORES_SEL = ['#c53030', '#dd6b20', '#1f4e79']
NOMES_SEL = [r'A (mín.\ $W_0$)', 'B (intermediária)', r'C (mín.\ $W_f$)']
_FUS_W = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]


class AnchorModel(oc.Model):
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


class EpsModel(AnchorModel):
    def __init__(self, eps_N, **kwargs):
        self.eps_N = float(eps_N)
        super(EpsModel, self).__init__(DV_NAMES, obj_key='W0', **kwargs)

    def confun_eps(self, x):
        res = self.results(x)
        g = oc.constraint_vector(res, self.con_names)
        return np.append(g, 1.0 - res['W_fuel']/self.eps_N)


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def solve_ancora(obj_key):
    model = AnchorModel(DV_NAMES, obj_key=obj_key)
    cons = [{'type': 'ineq', 'fun': model.confun, 'jac': model.conjac}]
    t0 = time.time()
    result = minimize(model.objfun, model.x0, jac=model.objgrad,
                      constraints=cons, bounds=model.bounds,
                      method='slsqp', options=OPTIONS_ANC)
    res = model.results(result.x)
    g = oc.constraint_vector(res)
    return model, result, time.time() - t0, res, g


def roda_ancoras():
    rows = []
    for caso, rotulo, b_max, w_max in CATEGORIAS:
        oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_max, w_max
        print('  ÂNCORAS %s (b_w < %.1f m)' % (rotulo, b_max))
        for obj_key, nome in [('W0', 'min W0'), ('W_fuel', 'min Wf')]:
            model, result, elapsed, res, g = solve_ancora(obj_key)
            x_phys = model.to_physical(result.x)
            ativas = [spec[0] for spec, gi in zip(CONSTRAINTS, g)
                      if abs(gi) <= TOL_ATIVA]
            print('    %s  W0=%.1f  Wf=%.1f kgf  %.2f s'
                  % (nome, res['W0']/gravity, res['W_fuel']/gravity, elapsed))
            row = {
                'caso': caso, 'rotulo': rotulo, 'ancora': nome, 'obj': obj_key,
                'b_w': res['b_w'], 'W0_kgf': res['W0']/gravity,
                'Wf_kgf': res['W_fuel']/gravity,
                'f1': res['W0']/model.res0['W0'],
                'f2': res['W_fuel']/model.res0['W_fuel'],
                'n_objfun': model.n_objfun, 'n_designTool': model.n_designTool,
                'tempo_s': elapsed, 'g_min': g.min(),
                'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
                'ativas': '; '.join(ativas), 'sucesso': bool(result.success),
            }
            for name, value in zip(DV_NAMES, x_phys):
                row[name] = value
            rows.append(row)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(RESULTS_DIR, 'moga_ancoras_todas.csv'),
                  index=False)
    df = df_all[df_all['caso'] == 'base'].set_index('ancora')
    df.to_csv(os.path.join(RESULTS_DIR, 'moga_ancoras.csv'))
    model_ref = AnchorModel(DV_NAMES, obj_key='W0')
    pd.DataFrame([{
        'W0_kgf': model_ref.res0['W0']/gravity,
        'Wf_kgf': model_ref.res0['W_fuel']/gravity,
    }]).to_csv(os.path.join(RESULTS_DIR, 'moga_referencia.csv'), index=False)
    return df_all, df


def populacao_inicial(model):
    rng = np.random.default_rng(SEED)
    n = len(model.x0)
    largura = model.xu - model.xl
    nuvem = (model.x0
             + DISPERSAO_SEMENTE*largura*rng.normal(size=(N_SEMEADOS - 1, n)))
    aleatorios = rng.uniform(model.xl, model.xu,
                             size=(POP_SIZE - N_SEMEADOS, n))
    return np.clip(np.vstack([model.x0.reshape(1, -1), nuvem, aleatorios]),
                   model.xl, model.xu)


def roda_moga(nome, caso):
    oc.B_W_MAX = caso['B_W_MAX']
    oc.WHEEL_SPAN_MAX = caso['WHEEL_SPAN_MAX']
    print('  MOGA %s  pop=%d  n_gen=%d' % (caso['rotulo'], POP_SIZE, N_GEN))
    model = MultiObjModel()
    problem = NJ0502BiObj(model=model)
    algorithm = NSGA2(
        pop_size=POP_SIZE, sampling=populacao_inicial(model),
        crossover=SBX(eta=ETA_CROSSOVER, prob=PROB_CROSSOVER),
        mutation=PM(eta=ETA_MUTATION), eliminate_duplicates=True)
    t0 = time.time()
    result = pymoo_minimize(problem, algorithm, get_termination('n_gen', N_GEN),
                            seed=SEED, save_history=True, verbose=False)
    elapsed = time.time() - t0
    n_eval = result.algorithm.evaluator.n_eval
    if result.X is None:
        print('  *** nenhuma solução viável ***')
        return None
    X, F = np.atleast_2d(result.X), np.atleast_2d(result.F)
    df = pd.DataFrame(frame_from_X(model, X, F)).sort_values('W0_kgf')
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'moga_frente_%s.csv' % nome),
              index=False)
    hist = []
    for gen, entry in enumerate(result.history, start=1):
        opt = entry.opt
        if opt is None or len(opt) == 0:
            continue
        Fg, Gg = opt.get('F'), opt.get('G')
        viavel = (Gg <= TOL_VIAVEL).all(axis=1) if Gg is not None else None
        hist.append({
            'geracao': gen, 'n_eval': entry.evaluator.n_eval,
            'n_frente': len(opt),
            'f1_min': float(Fg[:, 0].min()), 'f2_min': float(Fg[:, 1].min()),
            'W0_min_kgf': float(Fg[:, 0].min()*model.W0_ref/gravity),
            'Wf_min_kgf': float(Fg[:, 1].min()*model.Wf_ref/gravity),
            'n_viaveis': int(viavel.sum()) if viavel is not None else -1,
        })
    df_hist = pd.DataFrame(hist)
    df_hist.to_csv(os.path.join(RESULTS_DIR, 'moga_hist_%s.csv' % nome),
                   index=False)
    n_inviavel = int((~df['viavel']).sum())
    pd.DataFrame([{
        'caso': nome, 'rotulo': caso['rotulo'],
        'B_W_MAX': caso['B_W_MAX'], 'WHEEL_SPAN_MAX': caso['WHEEL_SPAN_MAX'],
        'pop_size': POP_SIZE, 'n_gen': N_GEN, 'seed': SEED,
        'n_semeados': N_SEMEADOS, 'n_eval': int(n_eval),
        'n_designTool': model.n_designTool, 'n_invalido': model.n_invalido,
        'tempo_s': elapsed, 'n_frente': len(df), 'n_inviavel': n_inviavel,
        'W0_min_kgf': df['W0_kgf'].min(), 'W0_max_kgf': df['W0_kgf'].max(),
        'Wf_min_kgf': df['Wf_kgf'].min(), 'Wf_max_kgf': df['Wf_kgf'].max(),
        'W0_ref_kgf': model.W0_ref/gravity,
        'Wf_ref_kgf': model.Wf_ref/gravity,
    }]).to_csv(os.path.join(RESULTS_DIR, 'moga_corrida_%s.csv' % nome),
               index=False)
    print('    n_eval=%d  %.1f s  W0 [%.1f, %.1f] kgf'
          % (n_eval, elapsed, df['W0_kgf'].min(), df['W0_kgf'].max()))
    return df


def frente_eps(caso, rotulo, b_max, w_max, ancoras):
    oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_max, w_max
    d = ancoras[ancoras['caso'] == caso].set_index('ancora')
    wf_lo = d.loc['min Wf', 'Wf_kgf']
    wf_hi = d.loc['min W0', 'Wf_kgf']
    print('  FRENTE REF %s  Wf [%.1f, %.1f] kgf' % (rotulo, wf_lo, wf_hi))
    rows = []
    for frac in np.linspace(0.0, 1.0, N_PONTOS):
        eps_kgf = wf_lo + frac*(wf_hi - wf_lo)
        model = EpsModel(eps_kgf*gravity)
        cons = [{'type': 'ineq', 'fun': model.confun_eps}]
        t0 = time.time()
        result = minimize(model.objfun, model.x0, constraints=cons,
                          bounds=model.bounds, method='slsqp',
                          options=OPTIONS_EPS)
        res = model.results(result.x)
        g = oc.constraint_vector(res)
        x_phys = model.to_physical(result.x)
        row = {
            'caso': caso, 'rotulo': rotulo, 'eps_kgf': eps_kgf,
            'W0_kgf': res['W0']/gravity, 'Wf_kgf': res['W_fuel']/gravity,
            'b_w': res['b_w'], 'g_min': g.min(),
            'viavel': bool(g.min() >= -1e-4), 'n_objfun': model.n_objfun,
            'tempo_s': time.time() - t0, 'sucesso': bool(result.success),
        }
        for name, value in zip(DV_NAMES, x_phys):
            row[name] = value
        rows.append(row)
    return rows


def seleciona_tres(df):
    d = df.sort_values('W0_kgf').reset_index(drop=True)
    return d.loc[[0, len(d)//2, len(d) - 1]].reset_index(drop=True)


def fig_pareto(df_moga, df_ref, anc, ref22, path, titulo, sel=None):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    for k, ax in enumerate(axes):
        ax.plot(df_ref['W0_kgf'], df_ref['Wf_kgf'], '-', color=COR_REF,
                lw=2.0, zorder=3,
                label=r'frente de referência ($\epsilon$-restrição + SLSQP)')
        ax.plot(df_moga['W0_kgf'], df_moga['Wf_kgf'], 'o', color=COR_MOGA,
                ms=4.5, zorder=4,
                label='frente do MOGA (%d pontos)' % len(df_moga))
        if sel is not None:
            for i, (_, row) in enumerate(sel.iterrows()):
                ax.plot(row['W0_kgf'], row['Wf_kgf'], '*', ms=17,
                        color=CORES_SEL[i], mec='white', mew=1.0, zorder=7,
                        label=NOMES_SEL[i] if k == 1 else None)
        ax.plot(anc.loc['min W0', 'W0_kgf'], anc.loc['min W0', 'Wf_kgf'],
                'X', ms=10, color=COR_ANC_W0, mec='white', mew=1.0, zorder=8,
                label=r'âncora SLSQP: mín $W_0$' if k == 0 else None)
        ax.plot(anc.loc['min Wf', 'W0_kgf'], anc.loc['min Wf', 'Wf_kgf'],
                'P', ms=10, color=COR_ANC_WF, mec='white', mew=1.0, zorder=8,
                label=r'âncora SLSQP: mín $W_f$' if k == 0 else None)
        ax.set_xlabel(r'$W_0$ [kgf]')
        ax.set_ylabel(r'$W_f$ [kgf]')
        ax.grid(color='0.92', lw=0.5)
    axes[0].plot(ref22['W0_kgf'], ref22['Wf_kgf'], 's', ms=9,
                 color=COR_PRJ22, mec='white', mew=1.0, zorder=8,
                 label='PRJ-22 (partida)')
    axes[0].set_title(titulo + '\n(escala do projeto)')
    axes[0].legend(fontsize=7.4, loc='upper left', framealpha=0.93)
    m0, M0 = df_ref['W0_kgf'].min(), df_ref['W0_kgf'].max()
    mf, Mf = df_ref['Wf_kgf'].min(), df_ref['Wf_kgf'].max()
    d0 = max(M0 - m0, 1e-6)*0.25
    df_ = max(Mf - mf, 1e-6)*0.25
    x0, x1, y0, y1 = m0 - d0, M0 + d0, mf - df_, Mf + df_
    axes[1].set_xlim(x0, x1)
    axes[1].set_ylim(y0, y1)
    axes[1].set_title('zoom na frente: %.1f kgf de amplitude em $W_0$'
                      % (M0 - m0))
    axes[1].ticklabel_format(useOffset=False, style='plain')
    axes[1].tick_params(labelsize=8)
    dentro = ((df_moga['W0_kgf'].between(x0, x1))
              & (df_moga['Wf_kgf'].between(y0, y1))).sum()
    manip, rotulos = axes[1].get_legend_handles_labels()
    if dentro == 0:
        pares = [(h, l) for h, l in zip(manip, rotulos)
                 if not l.startswith('frente do MOGA')]
        manip, rotulos = zip(*pares) if pares else ([], [])
        desvio = df_moga['W0_kgf'].min() - df_ref['W0_kgf'].min()
        axes[1].annotate(
            'os %d pontos do MOGA estão fora do quadro\n'
            r'($+%.0f$ kgf em $W_0$, %.0f$\times$ a largura da frente)'
            % (len(df_moga), desvio, desvio/max(M0 - m0, 1e-9)),
            xy=(0.5, 0.06), xycoords='axes fraction', ha='center',
            fontsize=7.8, color='#7b1d1d',
            bbox=dict(boxstyle='round,pad=0.35', fc='#fdf0f0',
                      ec='#c53030', lw=0.8))
    axes[1].legend(manip, rotulos, fontsize=7.4, loc='upper right',
                   framealpha=0.93)
    fig.tight_layout()
    _savefig(fig, path)


def fig_convergencia(hist, anc, path):
    W0_anc = anc.loc['min W0', 'W0_kgf']
    Wf_anc = anc.loc['min Wf', 'Wf_kgf']
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    axes[0].plot(hist['geracao'], hist['W0_min_kgf'], '-', lw=1.6,
                 color=COR_MOGA, label=r'melhor $W_0$ do MOGA')
    axes[0].axhline(W0_anc, color=COR_ANC_W0, lw=1.6, ls='--',
                    label=r'âncora SLSQP (Seção 3)')
    axes[0].set_ylabel(r'$W_0$ [kgf]')
    axes[0].set_title(r'Melhor $W_0$ por geração')
    axes[0].set_ylim(W0_anc*0.999, min(hist['W0_min_kgf'].max(), W0_anc*1.06))
    axes[0].legend(fontsize=8)
    gap_W0 = 100.0*(hist['W0_min_kgf']/W0_anc - 1.0)
    gap_Wf = 100.0*(hist['Wf_min_kgf']/Wf_anc - 1.0)
    axes[1].plot(hist['geracao'], gap_W0.clip(lower=1e-4), '-', lw=1.6,
                 color=COR_ANC_W0, label=r'$W_0$')
    axes[1].plot(hist['geracao'], gap_Wf.clip(lower=1e-4), '-', lw=1.6,
                 color=COR_ANC_WF, label=r'$W_f$')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('distância à âncora [\\%]')
    axes[1].set_title('Erro relativo às âncoras do SLSQP')
    axes[1].legend(fontsize=8)
    axes[2].plot(hist['geracao'], hist['n_frente'], '-', lw=1.6,
                 color=COR_MOGA, label='pontos na frente não dominada')
    if (hist['n_viaveis'] != hist['n_frente']).any():
        axes[2].plot(hist['geracao'], hist['n_viaveis'], '--', lw=1.4,
                     color='#2f855a', label='dos quais viáveis')
    axes[2].set_ylabel('indivíduos')
    axes[2].set_title('Tamanho da frente não dominada')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('geração')
        ax.grid(color='0.92', lw=0.5)
    fig.tight_layout()
    _savefig(fig, path)


def _fuselagem_planta(inputs):
    L_f, D_f, x_ts = inputs['L_f'], inputs['D_f'], inputs['x_tailstrike']
    xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_ts/L_f, 1.0])
    return xx*L_f, np.array(_FUS_W)*D_f


def _planta(ax, airplane, cor, rotulo, com_fuselagem, lw=1.7):
    geo, inp = airplane['geometry'], airplane['inputs']
    if com_fuselagem:
        xf, wf = _fuselagem_planta(inp)
        ax.plot(xf, wf/2, color='0.45', lw=1.0)
        ax.plot(xf, -wf/2, color='0.45', lw=1.0)
        for sy in (1.0, -1.0):
            r = 0.5*inp['D_n']
            x0, x1 = inp['x_n'], inp['x_n'] + inp['L_n']
            y0 = sy*inp['y_n']
            ax.plot([x0, x1, x1, x0, x0],
                    [y0 - r, y0 - r, y0 + r, y0 + r, y0 - r],
                    color='#d97706', lw=1.0)
    xw = [inp['xr_w'], geo['xt_w'], geo['xt_w'] + geo['ct_w'],
          inp['xr_w'] + geo['cr_w']]
    yw = [0.0, geo['yt_w'], geo['yt_w'], 0.0]
    ax.plot(xw + xw[::-1], yw + [-v for v in yw[::-1]], color=cor, lw=lw,
            label=rotulo)
    xh = [geo['xr_h'], geo['xt_h'], geo['xt_h'] + geo['ct_h'],
          geo['xr_h'] + geo['cr_h']]
    yh = [0.0, geo['yt_h'], geo['yt_h'], 0.0]
    ax.plot(xh + xh[::-1], yh + [-v for v in yh[::-1]], color=cor,
            lw=lw*0.75, ls='--')
    ax.plot([inp['x_mlg']]*2, [inp['y_mlg'], -inp['y_mlg']], 'o',
            color=cor, ms=4)


def fig_planformas(sel_base, sel_f, path):
    model = MultiObjModel()
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 9.8))
    painel = [
        (sel_base, 'Letra E (roteiro): as três aeronaves da frente '
                   'são indistinguíveis'),
        (sel_f, 'Letra F (teto relaxado): as três aeronaves diferem'),
    ]
    for ax, (sel, titulo) in zip(axes, painel):
        for i, (_, row) in enumerate(sel.iterrows()):
            inputs = model.build_inputs(
                np.array([row[n] for n in DV_NAMES])/model.scale)
            ap = run_designTool(inputs)
            rot = (r'%s: $b_w$=%.2f m, $W_0$=%.0f, $W_f$=%.0f kgf'
                   % (NOMES_SEL[i], ap['geometry']['b_w'],
                      row['W0_kgf'], row['Wf_kgf']))
            _planta(ax, ap, CORES_SEL[i], rot, com_fuselagem=(i == 0),
                    lw=2.0 - 0.45*i)
        ax.set_title(titulo)
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(color='0.92', lw=0.5)
        ax.legend(loc='lower right', fontsize=7.8)
    axes[1].set_xlabel('x [m]')
    fig.tight_layout()
    _savefig(fig, path)


def fig_ponta(sel_base, sel_f, path):
    model = MultiObjModel()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    painel = [(sel_base, 'Letra E: ponta da asa'),
              (sel_f, 'Letra F: ponta da asa')]
    for ax, (sel, titulo) in zip(axes, painel):
        for i, (_, row) in enumerate(sel.iterrows()):
            inputs = model.build_inputs(
                np.array([row[n] for n in DV_NAMES])/model.scale)
            ap = run_designTool(inputs)
            geo, inp = ap['geometry'], ap['inputs']
            xw = [inp['xr_w'], geo['xt_w'], geo['xt_w'] + geo['ct_w'],
                  inp['xr_w'] + geo['cr_w']]
            yw = [0.0, geo['yt_w'], geo['yt_w'], 0.0]
            ax.plot(xw, yw, color=CORES_SEL[i], lw=2.0 - 0.45*i,
                    label=r'%s: $b_w$=%.3f m' % (NOMES_SEL[i], geo['b_w']))
        ax.set_title(titulo)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(color='0.92', lw=0.5)
        ax.legend(fontsize=8, loc='lower left')
        ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    _savefig(fig, path)


def gera_figuras():
    anc = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_ancoras.csv'),
                      index_col=0)
    ref22 = pd.read_csv(os.path.join(RESULTS_DIR,
                                     'moga_referencia.csv')).iloc[0]
    df_ref_all = pd.read_csv(os.path.join(RESULTS_DIR, 'ref_frente_eps.csv'))
    fronts = {}
    for caso in ('base', 'letraF'):
        fronts[caso] = {
            'moga': pd.read_csv(os.path.join(
                RESULTS_DIR, 'moga_frente_%s.csv' % caso)
            ).sort_values('W0_kgf').reset_index(drop=True),
            'hist': pd.read_csv(os.path.join(
                RESULTS_DIR, 'moga_hist_%s.csv' % caso)),
            'ref': df_ref_all[df_ref_all['caso'] == caso
                              ].sort_values('W0_kgf').reset_index(drop=True),
        }
    sel_base = seleciona_tres(fronts['base']['ref'])
    sel_f = seleciona_tres(fronts['letraF']['ref'])
    sel_base.to_csv(os.path.join(RESULTS_DIR, 'sel_base.csv'), index=False)
    sel_f.to_csv(os.path.join(RESULTS_DIR, 'sel_letraF.csv'), index=False)
    fig_pareto(fronts['base']['moga'], fronts['base']['ref'], anc, ref22,
               os.path.join(RESULTS_DIR, 'moga_pareto_base.png'),
               'Frente de Pareto --- letra E (roteiro)', sel=sel_base)
    fig_pareto(fronts['letraF']['moga'], fronts['letraF']['ref'], anc, ref22,
               os.path.join(RESULTS_DIR, 'moga_pareto_letraF.png'),
               'Frente de Pareto --- letra F (teto relaxado)', sel=sel_f)
    fig_convergencia(fronts['base']['hist'], anc,
                     os.path.join(RESULTS_DIR, 'moga_convergencia.png'))
    fig_planformas(sel_base, sel_f,
                   os.path.join(RESULTS_DIR, 'moga_planformas.png'))
    fig_ponta(sel_base, sel_f, os.path.join(RESULTS_DIR, 'moga_ponta.png'))


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    b_ori, w_ori = oc.B_W_MAX, oc.WHEEL_SPAN_MAX
    try:
        print('=== âncoras ===')
        df_all, _ = roda_ancoras()
        print('=== MOGA ===')
        for nome in CASOS:
            roda_moga(nome, CASOS[nome])
        print('=== frente de referência ===')
        rows = []
        for caso, rotulo, b_max, w_max in CATEGORIAS:
            rows += frente_eps(caso, rotulo, b_max, w_max, df_all)
        pd.DataFrame(rows).to_csv(
            os.path.join(RESULTS_DIR, 'ref_frente_eps.csv'), index=False)
        print('=== figuras ===')
        gera_figuras()
    finally:
        oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_ori, w_ori
    print('  gravado em %s/' % RESULTS_DIR)
