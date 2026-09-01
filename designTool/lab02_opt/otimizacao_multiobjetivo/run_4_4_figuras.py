'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 4 - Grupo NJ-0502

Etapa 4 - Figuras da seção multiobjetivo.

  1. frente de Pareto do caso do roteiro (letra E), com a frente de
     referência da e-restrição, as âncoras do SLSQP e um zoom que mostra
     o quanto ela é estreita;
  2. convergência do MOGA por geração contra a âncora do Problema 3
     (resposta gráfica da pergunta 3);
  3. frente de Pareto com o teto de envergadura relaxado (letra F);
  4. vistas em planta de três aeronaves de regiões distintas da frente,
     nas duas categorias.

Uso:  python run_4_4_figuras.py   (depois de run_4_1, run_4_2 e run_4_3)
'''

# IMPORTS
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

from opt_multi import MultiObjModel, gravity
from opt_common import run_designTool

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_multiobjetivo')

COR_MOGA = '#1f4e79'
COR_REF = '#c53030'
COR_ANC_W0 = '#c53030'
COR_ANC_WF = '#2f855a'
COR_PRJ22 = '#6b46c1'

CORES_SEL = ['#c53030', '#dd6b20', '#1f4e79']
NOMES_SEL = [r'A (mín.\ $W_0$)', 'B (intermediária)', r'C (mín.\ $W_f$)']
NOMES_SEL_TXT = ['A (min W0)', 'B (intermediaria)', 'C (min Wf)']

DV = ['S_w', 'AR_w', 'sweep_w', 'xr_w', 'Cht', 'Cvt', 'x_mlg', 'y_mlg',
      'z_lg']


def _savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  gravado: %s' % os.path.basename(path))


def seleciona_tres(df):
    '''
    Três aeronaves de regiões distintas da frente: os dois extremos e o
    ponto do meio, ordenados por W0.
    '''
    d = df.sort_values('W0_kgf').reset_index(drop=True)
    return d.loc[[0, len(d)//2, len(d) - 1]].reset_index(drop=True)


# =========================================
# 1. FRENTES DE PARETO


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
                'X', ms=10, color=COR_ANC_W0, mec='white', mew=1.0,
                zorder=8, label=r'âncora SLSQP: mín $W_0$' if k == 0 else None)
        ax.plot(anc.loc['min Wf', 'W0_kgf'], anc.loc['min Wf', 'Wf_kgf'],
                'P', ms=10, color=COR_ANC_WF, mec='white', mew=1.0,
                zorder=8, label=r'âncora SLSQP: mín $W_f$' if k == 0 else None)

        ax.set_xlabel(r'$W_0$ [kgf]')
        ax.set_ylabel(r'$W_f$ [kgf]')
        ax.grid(color='0.92', lw=0.5)

    # Painel esquerdo: escala do projeto, com a partida de PRJ-22.
    # A legenda vai para o canto superior esquerdo, que é a região vazia:
    # os dados se acumulam embaixo à esquerda e PRJ-22 fica no topo à
    # direita.
    axes[0].plot(ref22['W0_kgf'], ref22['Wf_kgf'], 's', ms=9,
                 color=COR_PRJ22, mec='white', mew=1.0, zorder=8,
                 label='PRJ-22 (partida)')
    axes[0].set_title(titulo + '\n(escala do projeto)')
    axes[0].legend(fontsize=7.4, loc='upper left', framealpha=0.93)

    # Painel direito: zoom na frente de referência.
    m0, M0 = df_ref['W0_kgf'].min(), df_ref['W0_kgf'].max()
    mf, Mf = df_ref['Wf_kgf'].min(), df_ref['Wf_kgf'].max()
    d0 = max(M0 - m0, 1e-6)*0.25
    df_ = max(Mf - mf, 1e-6)*0.25
    x0, x1 = m0 - d0, M0 + d0
    y0, y1 = mf - df_, Mf + df_
    axes[1].set_xlim(x0, x1)
    axes[1].set_ylim(y0, y1)
    axes[1].set_title('zoom na frente: %.1f kgf de amplitude em $W_0$'
                      % (M0 - m0))
    axes[1].ticklabel_format(useOffset=False, style='plain')
    axes[1].tick_params(labelsize=8)

    # Nesta escala os pontos do MOGA quase sempre caem fora do quadro.
    # Dizer isso explicitamente evita que a legenda anuncie uma série
    # invisível como se ela estivesse ali.
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


# =========================================
# 2. CONVERGÊNCIA


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
    axes[0].set_ylim(W0_anc*0.999, min(hist['W0_min_kgf'].max(),
                                       W0_anc*1.06))
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

    # A frente não dominada só existe depois que aparece o primeiro
    # indivíduo viável; antes disso o NSGA-II ordena por violação.
    axes[2].plot(hist['geracao'], hist['n_frente'], '-', lw=1.6,
                 color=COR_MOGA, label='pontos na frente não dominada')
    viaveis = hist['n_viaveis']
    if (viaveis != hist['n_frente']).any():
        axes[2].plot(hist['geracao'], viaveis, '--', lw=1.4,
                     color='#2f855a', label='dos quais viáveis')
    axes[2].set_ylabel('indivíduos')
    axes[2].set_title('Tamanho da frente não dominada')
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel('geração')
        ax.grid(color='0.92', lw=0.5)

    fig.tight_layout()
    _savefig(fig, path)


# =========================================
# 3. VISTAS EM PLANTA


_FUS_W = [0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]


def _fuselagem_planta(inputs):
    L_f, D_f = inputs['L_f'], inputs['D_f']
    x_ts = inputs['x_tailstrike']
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
                np.array([row[n] for n in DV])/model.scale)
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
    '''
    Zoom na ponta da asa: é onde as três aeronaves de cada frente se
    separam (ou não). Deixa visível que na letra E elas coincidem.
    '''
    model = MultiObjModel()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    painel = [(sel_base, 'Letra E: ponta da asa'),
              (sel_f, 'Letra F: ponta da asa')]

    for ax, (sel, titulo) in zip(axes, painel):
        for i, (_, row) in enumerate(sel.iterrows()):
            inputs = model.build_inputs(
                np.array([row[n] for n in DV])/model.scale)
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


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    anc = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_ancoras.csv'),
                      index_col=0)
    ref22 = pd.read_csv(os.path.join(RESULTS_DIR,
                                     'moga_referencia.csv')).iloc[0]
    df_ref_all = pd.read_csv(os.path.join(RESULTS_DIR,
                                          'ref_frente_eps.csv'))

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

    # As três aeronaves saem da frente de REFERÊNCIA, que é a convergida.
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

    fig_ponta(sel_base, sel_f,
              os.path.join(RESULTS_DIR, 'moga_ponta.png'))

    print('\n  Arquivos gravados em %s/' % RESULTS_DIR)
