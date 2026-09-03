'''
PRJ-23 Lab 02 - Problema 3. SLSQP na NJ-0502 e figuras.
Uso: python run_3_otimizacao.py
'''

import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
from scipy.optimize import minimize

from opt_common import (CONSTRAINTS, DESIGN_VARS, DV_NAMES, Model,
                        constraint_vector, get_baseline, gravity,
                        physical_report, run_designTool)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_NJ0502')
OPTIONS = {'maxiter': 200, 'ftol': 1e-6, 'disp': False}
TOL_ATIVA = 1e-4
COR_BASE, COR_OPT = '#1f4e79', '#c53030'
XLABEL_NF = r'chamada $n_f$ da função objetivo'
CORES_G = [
    '#c53030', '#2b6cb0', '#2f855a', '#dd6b20', '#6b46c1',
    '#0d9488', '#b83280', '#1a365d', '#276749', '#9c4221',
    '#553c9a', '#234e52', '#744210', '#9b2c2c', '#2c5282',
    '#22543d', '#7b341e', '#4a5568', '#718096',
]
GRUPO_G = [
    ('desempenho e pátio', 'opt_hist_g_desempenho.png',
     ['landing', 'tank', 'thrust', 'span', 'wheelspan', 'height', 'CLv',
      'vt_te', 'ht_te']),
    ('estabilidade e trem', 'opt_hist_g_estabilidade.png',
     ['SM_fwd', 'SM_aft', 'SM_aft_max', 'nlg_fwd', 'nlg_aft',
      'tipback', 'tailstrike', 'overturn', 'gear_te', 'gear_spar']),
]
_FUS_H = [0.0, 2.27/4.0, 3.56/4.0, 1.0, 1.0, 1.07/4.0]
_FUS_W = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]


def run_opt():
    model = Model(DV_NAMES)
    cons = [{'type': 'ineq', 'fun': model.confun, 'jac': model.conjac}]
    t0 = time.time()
    result = minimize(model.objfun, model.x0, jac=model.objgrad,
                      constraints=cons, bounds=model.bounds,
                      method='slsqp', options=OPTIONS)
    return model, result, time.time() - t0


def history_frame(model):
    data = {'f': model.hist_f}
    x_hist = np.array(model.hist_x)
    for j, name in enumerate(model.dv_names):
        data[name] = x_hist[:, j]*model.scale[j]
    g_hist = np.array(model.hist_g)
    for j, spec in enumerate(CONSTRAINTS):
        data['g_' + spec[0]] = g_hist[:, j]
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = 'n_f'
    return df


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def fig_variaveis(df, path):
    specs = {s[0]: s for s in DESIGN_VARS}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name in DV_NAMES:
        serie = df[name]
        ax.plot(df.index, serie/serie.iloc[0], 'o-', ms=5, lw=1.4,
                label=specs[name][1])
    ax.axhline(1.0, color='gray', lw=0.6)
    ax.set_ylabel(r'$x/x_{\mathrm{inicial}}$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico das variáveis de projeto')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_xticks(list(df.index))
    _savefig(fig, path)


def fig_objetivo(df, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(df.index, df['f'], 'o-', ms=6, lw=1.8, color=COR_OPT)
    ax.set_ylabel(r'$f = W_0/W_{0,\mathrm{ref}}$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico da função objetivo')
    ax.set_xticks(list(df.index))
    _savefig(fig, path)


def fig_restricoes(df, path, titulo, nomes, cor0):
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    k = cor0
    for name in nomes:
        col = 'g_' + name
        if col not in df.columns:
            continue
        if df[col].std() < 1e-10 and abs(df[col].iloc[-1]) > 5e-2:
            continue
        ax.plot(df.index, df[col], 'o-', ms=5, lw=1.4,
                color=CORES_G[k % len(CORES_G)], label=name)
        k += 1
    ax.axhline(0.0, color='gray', lw=0.9)
    ax.set_ylabel(r'$g$')
    ax.set_xlabel(XLABEL_NF)
    ax.set_title('Histórico das restrições --- %s' % titulo)
    ax.set_ylim(-0.25, 1.35)
    ax.set_xticks(list(df.index))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _savefig(fig, path)
    return k


def _fuselagem(inputs):
    L_f, D_f, x_ts = inputs['L_f'], inputs['D_f'], inputs['x_tailstrike']
    xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_ts/L_f, 1.0])
    return xx*L_f, np.array(_FUS_H)*D_f, np.array(_FUS_W)*D_f


def _nacele_cilindro(ax, x0, y0, length, diam, cor, n_est=11):
    r = 0.5*diam
    x1 = x0 + length
    ax.plot([x0, x1, x1, x0, x0],
            [y0 - r, y0 - r, y0 + r, y0 + r, y0 - r],
            color=cor, lw=1.2, zorder=3)
    for x in np.linspace(x0, x1, n_est):
        ax.plot([x, x], [y0 - r, y0 + r], color=cor, lw=0.7, zorder=3)
    ax.plot([x0, x1], [y0, y0], color=cor, lw=0.8, zorder=3)


def _planta(ax, airplane, cor, rotulo, com_fuselagem):
    geo, inp = airplane['geometry'], airplane['inputs']
    if com_fuselagem:
        xf, _, wf = _fuselagem(inp)
        ax.plot(xf, wf/2, color='0.4', lw=1.0)
        ax.plot(xf, -wf/2, color='0.4', lw=1.0)
        for sy in (1.0, -1.0):
            _nacele_cilindro(ax, inp['x_n'], sy*inp['y_n'],
                             inp['L_n'], inp['D_n'], '#d97706')
    xw = [inp['xr_w'], geo['xt_w'], geo['xt_w'] + geo['ct_w'],
          inp['xr_w'] + geo['cr_w']]
    yw = [0.0, geo['yt_w'], geo['yt_w'], 0.0]
    ax.plot(xw + xw[::-1], yw + [-v for v in yw[::-1]], color=cor, lw=1.6,
            label=rotulo)
    xh = [geo['xr_h'], geo['xt_h'], geo['xt_h'] + geo['ct_h'],
          geo['xr_h'] + geo['cr_h']]
    yh = [0.0, geo['yt_h'], geo['yt_h'], 0.0]
    ax.plot(xh + xh[::-1], yh + [-v for v in yh[::-1]], color=cor, lw=1.2,
            ls='--')
    ax.plot([inp['x_mlg']]*2, [inp['y_mlg'], -inp['y_mlg']], 'o',
            color=cor, ms=4)


def _lateral(ax, airplane, cor, rotulo, com_fuselagem):
    geo, inp = airplane['geometry'], airplane['inputs']
    if com_fuselagem:
        xf, hf, _ = _fuselagem(inp)
        x_ts = inp['x_tailstrike']
        desloc = np.where(xf > x_ts, (inp['D_f'] - hf)/2, 0.0)
        z_top = hf/2 + desloc
        z_bot = -hf/2 + desloc
        ax.plot(xf, z_top, color='0.4', lw=1.0)
        ax.plot(xf, z_bot, color='0.4', lw=1.0)
        ax.plot([xf[-1], xf[-1]], [z_top[-1], z_bot[-1]], color='0.4', lw=1.0)
        _nacele_cilindro(ax, inp['x_n'], inp['z_n'],
                         inp['L_n'], inp['D_n'], '#d97706')
    ax.plot([inp['xr_w'], inp['xr_w'] + geo['cr_w']],
            [inp['zr_w']]*2, color=cor, lw=2.4, label=rotulo)
    ax.plot([geo['xr_h'], geo['xr_h'] + geo['cr_h']],
            [inp['zr_h']]*2, color=cor, lw=1.8, ls='--')
    xv = [geo['xr_v'], geo['xt_v'], geo['xt_v'] + geo['ct_v'],
          geo['xr_v'] + geo['cr_v'], geo['xr_v']]
    zv = [inp['zr_v'], geo['zt_v'], geo['zt_v'], inp['zr_v'], inp['zr_v']]
    ax.plot(xv, zv, color=cor, lw=1.4)
    ax.plot([inp['x_nlg'], inp['x_mlg']], [inp['z_lg']]*2, 'o',
            color=cor, ms=5)
    ax.plot([0.0, inp['L_f']], [inp['z_lg']]*2, color=cor, lw=0.7, ls=':')
    ang = np.arctan((inp['z_tailstrike'] - inp['z_lg'])
                    / (inp['x_tailstrike'] - inp['x_mlg']))
    ax.plot([inp['x_mlg'], inp['L_f']],
            [inp['z_lg'], inp['z_lg'] + (inp['L_f'] - inp['x_mlg'])*np.tan(ang)],
            color=cor, lw=0.7, ls='-.')


def _xmax_aeronave(airplane):
    geo, inp = airplane['geometry'], airplane['inputs']
    return max(inp['L_f'],
               geo['xr_v'] + geo['cr_v'], geo['xt_v'] + geo['ct_v'],
               geo['xr_h'] + geo['cr_h'], geo['xt_h'] + geo['ct_h'])


def vistas(inputs_base, inputs_opt, path):
    ap_base = run_designTool(inputs_base)
    ap_opt = run_designTool(inputs_opt)
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 8.2),
                             gridspec_kw={'height_ratios': [2.15, 1.0]})
    _planta(axes[0], ap_base, COR_BASE, 'PRJ-22 (partida)', True)
    _planta(axes[0], ap_opt, COR_OPT, 'otimizada', False)
    axes[0].set_ylabel('y [m]')
    axes[0].set_title('Vista em planta')
    axes[0].legend(loc='upper right', fontsize=9)
    _lateral(axes[1], ap_base, COR_BASE, 'PRJ-22 (partida)', True)
    _lateral(axes[1], ap_opt, COR_OPT, 'otimizada', False)
    axes[1].set_ylabel('z [m]')
    axes[1].set_xlabel('x [m]')
    axes[1].set_title('Vista lateral')
    axes[1].legend(loc='upper right', fontsize=9)
    pad = 1.8
    x_max = max(_xmax_aeronave(ap_base), _xmax_aeronave(ap_opt))
    b_max = max(ap_base['geometry']['b_w'], ap_opt['geometry']['b_w'])
    z_lo = min(ap_base['inputs']['z_lg'], ap_opt['inputs']['z_lg'])
    z_hi = max(ap_base['geometry']['zt_v'], ap_opt['geometry']['zt_v'])
    axes[0].set_xlim(-pad, x_max + pad)
    axes[0].set_ylim(-b_max/2.0 - pad, b_max/2.0 + pad)
    axes[1].set_xlim(-pad, x_max + pad)
    axes[1].set_ylim(z_lo - pad, z_hi + pad)
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='0.9', lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def grava_csv(model, result, elapsed, res, g, x_phys, ganho, df_hist):
    df_hist.to_csv(os.path.join(RESULTS_DIR, 'opt_hist.csv'))
    pd.DataFrame([{
        'sucesso': bool(result.success), 'status': result.message,
        'n_vars': len(model.dv_names), 'n_iter': int(result.nit),
        'n_objfun': model.n_objfun, 'n_designTool': model.n_designTool,
        'tempo_s': elapsed, 'W0_kgf': res['W0']/gravity,
        'f': res['W0']/model.W0_ref, 'ganho_pct': ganho, 'g_min': g.min(),
        'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
    }]).to_csv(os.path.join(RESULTS_DIR, 'opt_corrida.csv'), index=False)
    rows_dv = []
    for name, x0v, xov in zip(DV_NAMES, model.to_physical(model.x0), x_phys):
        spec = [s for s in DESIGN_VARS if s[0] == name][0]
        fator = spec[3]
        rows_dv.append({
            'variavel': name, 'label': spec[1], 'unidade': spec[2],
            'inicial': x0v*fator, 'otimo': xov*fator,
            'lim_inf': spec[4]*fator, 'lim_sup': spec[5]*fator,
            'variacao_pct': 100.0*(xov/x0v - 1.0),
            'inicial_si': x0v, 'otimo_si': xov})
    df_dv = pd.DataFrame(rows_dv).set_index('variavel')
    df_dv.to_csv(os.path.join(RESULTS_DIR, 'opt_variaveis.csv'))
    g_base = constraint_vector(model.res0)
    rows_con = []
    for spec, gb, go in zip(CONSTRAINTS, g_base, g):
        rows_con.append({
            'restricao': spec[0], 'descricao': spec[1], 'expressao': spec[2],
            'origem': spec[4], 'g_inicial': gb, 'g_otimo': go,
            'ativa': bool(abs(go) <= TOL_ATIVA)})
    pd.DataFrame(rows_con).set_index('restricao').to_csv(
        os.path.join(RESULTS_DIR, 'opt_restricoes.csv'))
    rows_phys = []
    for (label, v_base), (_, v_opt) in zip(physical_report(model.res0),
                                           physical_report(res)):
        rows_phys.append({
            'grandeza': label, 'inicial': v_base, 'otimo': v_opt,
            'variacao_pct': (100.0*(v_opt/v_base - 1.0)
                             if abs(v_base) > 1e-12 else np.nan)})
    pd.DataFrame(rows_phys).set_index('grandeza').to_csv(
        os.path.join(RESULTS_DIR, 'opt_grandezas.csv'))
    return df_dv


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('  OTIMIZAÇÃO NJ-0502  |  %s' % ', '.join(DV_NAMES))
    model, result, elapsed = run_opt()
    res = model.results(result.x)
    g = constraint_vector(res)
    x_phys = model.to_physical(result.x)
    ganho = 100.0*(1.0 - res['W0']/model.W0_ref)
    print('  %s | n_f=%d | n_dt=%d | %.2f s | W0=%.1f kgf (%.2f%%)'
          % (result.message, model.n_objfun, model.n_designTool, elapsed,
             res['W0']/gravity, ganho))
    df_hist = history_frame(model)
    df_dv = grava_csv(model, result, elapsed, res, g, x_phys, ganho, df_hist)
    inputs_base, inputs_opt = get_baseline(), get_baseline()
    for name, row in df_dv.iterrows():
        inputs_opt[name] = row['otimo_si']
    vistas(inputs_base, inputs_opt, os.path.join(RESULTS_DIR, 'opt_vistas.png'))
    fig_variaveis(df_hist, os.path.join(RESULTS_DIR, 'opt_hist_variaveis.png'))
    fig_objetivo(df_hist, os.path.join(RESULTS_DIR, 'opt_hist_objetivo.png'))
    k = 0
    for titulo, fname, nomes in GRUPO_G:
        k = fig_restricoes(df_hist, os.path.join(RESULTS_DIR, fname),
                           titulo, nomes, k)
    print('  gravado em %s/' % RESULTS_DIR)
