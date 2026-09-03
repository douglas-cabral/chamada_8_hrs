'''
PRJ-23 Lab 02 - Problema 2. SLSQP no Fokker 100 e figuras.
Uso: python run_2_otimizacao.py
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

from opt_padrao import (B_W_MAX, CONSTRAINTS, DESIGN_VARS, DV_NAMES, Model,
                        constraint_vector, get_baseline, gravity,
                        physical_report, run_designTool)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'resultados_otimizacao_aeronave_padrao')
OPTIONS = {'maxiter': 200, 'ftol': 1e-8, 'disp': False}
TOL_ATIVA = 1e-4
COR_BASE = '#1f4e79'
COR_OPT = '#c53030'
N_AR, N_SW = 61, 61
_FUS_W = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]


def run_opt():
    model = Model()
    t0 = time.time()
    result = minimize(model.objfun, model.x0,
                      constraints=[{'type': 'ineq', 'fun': model.confun}],
                      bounds=model.bounds, method='slsqp', options=OPTIONS)
    return model, result, time.time() - t0


def history_frame(model):
    data = {'f': model.hist_f}
    x_hist = np.array(model.hist_x)
    for j, name in enumerate(model.dv_names):
        data[name] = x_hist[:, j]*model.scale[j]
    g_hist = np.array(model.hist_g)
    for j, spec in enumerate(CONSTRAINTS):
        data['g_' + spec[0]] = g_hist[:, j]
    data['b_w'] = (1.0 - g_hist[:, 0])*B_W_MAX
    df = pd.DataFrame(data)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = 'n_f'
    return df


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def malha_W0(path_csv):
    if os.path.isfile(path_csv):
        print('  malha lida do cache: %s' % os.path.basename(path_csv))
        return pd.read_csv(path_csv)
    print('  calculando malha %dx%d de W0...' % (N_AR, N_SW))
    mod = Model()
    AR_lim = [s for s in DESIGN_VARS if s[0] == 'AR_w'][0][4:6]
    SW_lim = [s for s in DESIGN_VARS if s[0] == 'S_w'][0][4:6]
    rows = []
    for AR in np.linspace(AR_lim[0], AR_lim[1], N_AR):
        for S in np.linspace(SW_lim[0], SW_lim[1], N_SW):
            res = mod.results(np.array([AR/mod.scale[0], S/mod.scale[1]]))
            rows.append({'AR_w': AR, 'S_w': S,
                         'W0_kgf': res['W0']/gravity, 'b_w': res['b_w']})
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    return df


def fig_mapa(df_malha, df_hist, df_tab1, path):
    ARs = np.unique(df_malha['AR_w'].values)
    SWs = np.unique(df_malha['S_w'].values)
    W0 = df_malha['W0_kgf'].values.reshape(len(ARs), len(SWs)).T
    AA, SS = np.meshgrid(ARs, SWs)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    niveis = np.arange(40500, 50001, 500)
    cf = ax.contourf(AA, SS, W0, levels=niveis, cmap='viridis_r', extend='both')
    cs = ax.contour(AA, SS, W0, levels=niveis, colors='white',
                    linewidths=0.4, alpha=0.6)
    ax.clabel(cs, levels=niveis[::2], fmt='%.0f', fontsize=6, colors='white')
    fig.colorbar(cf, ax=ax, pad=0.02).set_label(r'$W_0$ [kgf]')
    B_w = df_malha['b_w'].values.reshape(len(ARs), len(SWs)).T
    ax.contourf(AA, SS, B_w, levels=[B_W_MAX, 1e9], colors=['#c53030'],
                alpha=0.22)
    ax.contour(AA, SS, B_w, levels=[B_W_MAX], colors=['#c53030'],
               linewidths=2.2)
    ax.plot([], [], color='#c53030', lw=2.2, label=r'$b_w = 30$ m (fronteira)')
    ax.text(10.55, 114.0, 'inviável\n' + r'$b_w > 30$ m', color='#7b1d1d',
            fontsize=9, ha='center', va='center', weight='bold')
    ax.plot(df_hist['AR_w'], df_hist['S_w'], '-', color='white', lw=1.0,
            alpha=0.8, zorder=4)
    ax.plot(df_hist['AR_w'], df_hist['S_w'], 'o', color='white', ms=2.6,
            zorder=5)
    ini, oti = df_tab1.loc['Partida'], df_tab1.loc['Ótimo']
    ax.plot(ini['AR_w'], ini['S_w'], 'o', color=COR_BASE, ms=10,
            mec='white', mew=1.4, zorder=6, label='partida (7,5; 90)')
    ax.plot(oti['AR_w'], oti['S_w'], '*', color=COR_OPT, ms=20,
            mec='white', mew=1.2, zorder=6, label=r'ótimo (10,47; 85,0)')
    ax.set_xlabel(r'$AR_w$ [--]')
    ax.set_ylabel(r'$S_w$ [m$^2$]')
    ax.set_title(r'Mapa de $W_0$ na caixa de projeto e trajetória do SLSQP')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.92)
    _savefig(fig, path)


def fig_historico(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    axes[0].plot(df.index, df['f'], 'o-', ms=4, lw=1.6, color=COR_OPT)
    axes[0].axhline(1.0, color='gray', lw=0.6)
    axes[0].set_ylabel(r'$f = W_0/W_{0,\mathrm{ref}}$')
    axes[0].set_xlabel(r'chamada $n_f$ da função objetivo')
    axes[0].set_title('Função objetivo')
    specs = {s[0]: s for s in DESIGN_VARS}
    for name in DV_NAMES:
        serie = df[name]
        axes[1].plot(df.index, serie/serie.iloc[0], 'o-', ms=4, lw=1.5,
                     label=specs[name][1])
    axes[1].plot(df.index, df['b_w']/df['b_w'].iloc[0], 's--', ms=4, lw=1.3,
                 color='#2f855a', label=r'$b_w$')
    axes[1].axhline(1.0, color='gray', lw=0.6)
    axes[1].set_ylabel(r'$x/x_{\mathrm{inicial}}$')
    axes[1].set_xlabel(r'chamada $n_f$ da função objetivo')
    axes[1].set_title('Variáveis de projeto e envergadura')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(color='0.92', lw=0.5)
    fig.tight_layout()
    _savefig(fig, path)


def _fuselagem_planta(inputs):
    L_f, D_f, x_ts = inputs['L_f'], inputs['D_f'], inputs['x_tailstrike']
    xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, x_ts/L_f, 1.0])
    return xx*L_f, np.array(_FUS_W)*D_f


def _planta(ax, airplane, cor, rotulo, com_fuselagem):
    geo, inp = airplane['geometry'], airplane['inputs']
    if com_fuselagem:
        xf, wf = _fuselagem_planta(inp)
        ax.plot(xf, wf/2, color='0.45', lw=1.0)
        ax.plot(xf, -wf/2, color='0.45', lw=1.0)
    xw = [inp['xr_w'], geo['xt_w'], geo['xt_w'] + geo['ct_w'],
          inp['xr_w'] + geo['cr_w']]
    yw = [0.0, geo['yt_w'], geo['yt_w'], 0.0]
    ax.plot(xw + xw[::-1], yw + [-v for v in yw[::-1]], color=cor, lw=1.7,
            label=rotulo)
    xh = [geo['xr_h'], geo['xt_h'], geo['xt_h'] + geo['ct_h'],
          geo['xr_h'] + geo['cr_h']]
    yh = [0.0, geo['yt_h'], geo['yt_h'], 0.0]
    ax.plot(xh + xh[::-1], yh + [-v for v in yh[::-1]], color=cor, lw=1.2,
            ls='--')


def fig_vistas(inputs_base, inputs_opt, path):
    ap_base = run_designTool(inputs_base)
    ap_opt = run_designTool(inputs_opt)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _planta(ax, ap_base, COR_BASE,
            r'partida: $AR_w=7{,}5$, $S_w=90$ m$^2$', True)
    _planta(ax, ap_opt, COR_OPT,
            r'ótimo: $AR_w=10{,}47$, $S_w=85{,}0$ m$^2$', False)
    for sy in (1, -1):
        ax.axhline(sy*B_W_MAX/2, color='#c53030', lw=0.9, ls=':')
    ax.text(1.0, B_W_MAX/2 + 0.4, r'$b_w/2 = 15$ m (teto)',
            color='#c53030', fontsize=8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Vista em planta: asa de partida e asa otimizada')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(color='0.92', lw=0.5)
    fig.tight_layout()
    _savefig(fig, path)


def grava_csv(model, result, elapsed, res, res0, g, x_phys, ganho, df_hist):
    df_hist.to_csv(os.path.join(RESULTS_DIR, 'opt_hist.csv'))
    df_tab1 = pd.DataFrame([
        {'ponto': 'Partida', 'AR_w': model.x0_phys[0], 'S_w': model.x0_phys[1],
         'MTOW_kgf': res0['W0']/gravity, 'MTOW_N': res0['W0'],
         'b_w': res0['b_w']},
        {'ponto': 'Ótimo', 'AR_w': x_phys[0], 'S_w': x_phys[1],
         'MTOW_kgf': res['W0']/gravity, 'MTOW_N': res['W0'],
         'b_w': res['b_w']},
    ]).set_index('ponto')
    df_tab1.to_csv(os.path.join(RESULTS_DIR, 'opt_tabela1.csv'))
    pd.DataFrame([{
        'sucesso': bool(result.success), 'status': result.message,
        'n_vars': len(model.dv_names), 'n_iter': int(result.nit),
        'nfev_scipy': int(result.nfev), 'njev_scipy': int(result.njev),
        'n_objfun': model.n_objfun, 'n_confun': model.n_confun,
        'n_designTool': model.n_designTool, 'tempo_s': elapsed,
        'W0_ini_kgf': res0['W0']/gravity, 'W0_kgf': res['W0']/gravity,
        'f': res['W0']/model.W0_ref, 'ganho_pct': ganho, 'b_w': res['b_w'],
        'g_span': g[0], 'g_min': g.min(),
        'n_ativas': int((np.abs(g) <= TOL_ATIVA).sum()),
    }]).to_csv(os.path.join(RESULTS_DIR, 'opt_corrida.csv'), index=False)
    rows_dv = []
    for name, x0v, xov in zip(DV_NAMES, model.x0_phys, x_phys):
        spec = [s for s in DESIGN_VARS if s[0] == name][0]
        rows_dv.append({
            'variavel': name, 'label': spec[1], 'unidade': spec[2],
            'inicial': x0v, 'otimo': xov, 'lim_inf': spec[4],
            'lim_sup': spec[5], 'variacao_pct': 100.0*(xov/x0v - 1.0),
            'no_limite': bool(
                abs(xov - spec[4]) < 1e-6*max(1.0, abs(spec[4]))
                or abs(xov - spec[5]) < 1e-6*max(1.0, abs(spec[5])))})
    pd.DataFrame(rows_dv).set_index('variavel').to_csv(
        os.path.join(RESULTS_DIR, 'opt_variaveis.csv'))
    g_base = constraint_vector(res0)
    rows_con = []
    for spec, gb, go in zip(CONSTRAINTS, g_base, g):
        rows_con.append({
            'restricao': spec[0], 'descricao': spec[1], 'expressao': spec[2],
            'origem': spec[4], 'g_inicial': gb, 'g_otimo': go,
            'ativa': bool(abs(go) <= TOL_ATIVA)})
    pd.DataFrame(rows_con).set_index('restricao').to_csv(
        os.path.join(RESULTS_DIR, 'opt_restricoes.csv'))
    rows_phys = []
    for (label, v_base), (_, v_opt) in zip(physical_report(res0),
                                           physical_report(res)):
        rows_phys.append({
            'grandeza': label, 'inicial': v_base, 'otimo': v_opt,
            'variacao_pct': (100.0*(v_opt/v_base - 1.0)
                             if abs(v_base) > 1e-12 else np.nan)})
    pd.DataFrame(rows_phys).set_index('grandeza').to_csv(
        os.path.join(RESULTS_DIR, 'opt_grandezas.csv'))
    return df_tab1


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('  OTIMIZAÇÃO DA AERONAVE PADRÃO  |  %s' % ', '.join(DV_NAMES))
    model, result, elapsed = run_opt()
    res, res0 = model.results(result.x), model.res0
    g = constraint_vector(res)
    x_phys = model.to_physical(result.x)
    ganho = 100.0*(1.0 - res['W0']/model.W0_ref)
    print('  %s | nfev=%d | %.3f s | W0=%.2f kgf (%.3f%%) | b_w=%.4f m'
          % (result.message, result.nfev, elapsed, res['W0']/gravity, ganho,
             res['b_w']))
    df_hist = history_frame(model)
    df_tab1 = grava_csv(model, result, elapsed, res, res0, g, x_phys, ganho,
                        df_hist)
    df_malha = malha_W0(os.path.join(RESULTS_DIR, 'mapa_W0.csv'))
    fig_mapa(df_malha, df_hist, df_tab1,
             os.path.join(RESULTS_DIR, 'opt_mapa.png'))
    fig_historico(df_hist, os.path.join(RESULTS_DIR, 'opt_historico.png'))
    inputs_base, inputs_opt = get_baseline(), get_baseline()
    inputs_base['AR_w'] = df_tab1.loc['Partida', 'AR_w']
    inputs_base['S_w'] = df_tab1.loc['Partida', 'S_w']
    inputs_opt['AR_w'] = df_tab1.loc['Ótimo', 'AR_w']
    inputs_opt['S_w'] = df_tab1.loc['Ótimo', 'S_w']
    fig_vistas(inputs_base, inputs_opt,
               os.path.join(RESULTS_DIR, 'opt_vistas.png'))
    print('  gravado em %s/' % RESULTS_DIR)
