'''
PRJ-23 Lab 04 - Item 2.4: metodo da secao critica.

Determina o CLmax de asa limpa em baixa velocidade para as duas posicoes de CG,
com e sem a restricao de trimagem, e grava as distribuicoes de sustentacao na
condicao de estol.

Condicao de voo
---------------
V2 = 1,2 Vs de decolagem no nivel do mar (M = 0,25, Re_MAC = 4,0e7). E o mesmo
par Re/M em que o cl_max do perfil foi levantado no XFOIL durante o Lab 03, de
modo que o limite usado aqui e o mesmo daquele relatorio. Todas as superficies
de alta sustentacao ficam neutras (asa limpa); o unico comando ativo e o
profundor, nos casos trimados.

Criterio de estol
-----------------
O comando fs do AVL devolve, faixa por faixa, duas colunas de coeficiente de
sustentacao da secao:

  cl       : referido a corda da corrente livre e a pressao dinamica do
             escoamento livre;
  cl_norm  : o mesmo carregamento referido ao plano normal ao bordo de ataque,
             isto e, cl_norm = cl/cos^2(Lambda) (verificado numericamente: a
             razao entre as colunas e praticamente constante ao longo da
             envergadura, ~1,500, contra 1/cos^2(35,2 graus) = 1,4975; a
             diferenca de 0,2% vem do diedro).

O roteiro do enunciado (passo 14 da Tab. 3) manda comparar cl_norm com o cl_max
do perfil, e e esse o criterio adotado como resultado oficial. Como a leitura
do limite muda bastante o CLmax, o script avalia tres criterios e reporta os
tres:

  roteiro   cl_norm  x  cl_max do perfil da corrente livre, no Re local
            (resultado oficial);
  normal    cl_norm  x  cl_max do perfil do plano normal (t/c esticado por
            1/cos(Lambda)) em M_n = M cos(Lambda) e Re_n = Re cos(Lambda) - a
            versao internamente coerente da teoria de enflechamento simples;
  corrente  cl       x  cl_max do perfil da corrente livre, no Re local -
            leitura sem correcao de enflechamento, que serve de limite superior.

Em todos eles o cl_max do perfil varia com a envergadura, porque a corda cai de
10,02 m na raiz a 2,00 m na ponta (Re local variando por um fator de 5).

Implementacao
-------------
Para cada caso roda-se uma unica sessao do AVL com uma varredura de alpha de
4 a 20 graus; a razao max(cl/cl_max) e interpolada linearmente para achar o
alpha de estol de cada criterio, e o AVL e reexecutado nesse alpha para ler
CLmax e delta_e.

Uso (a partir desta pasta):
    python q4_secao_critica.py           # roda o AVL e grava tudo
    python q4_secao_critica.py --tex     # refaz so CSV/tabelas/figuras
'''

import csv
import json
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.abspath(__file__))
for p in (LAB, os.path.abspath(os.path.join(LAB, '..')),
          os.path.abspath(os.path.join(LAB, '..', 'lab02_opt',
                                       'otimizacao_NJ0502'))):
    if p not in sys.path:
        sys.path.insert(0, p)

import avl_tools as at                     # noqa: E402
import ponto_projeto as pj                 # noqa: E402
import xfoil_perfil as xp                  # noqa: E402
import opt_common as oc                    # noqa: E402

RESCOM = os.path.join(LAB, 'resultados_comum')
OUT = os.path.join(LAB, 'resultados_q4')
TEX = os.path.join(LAB, 'tex_q4')

CASOS = [('fwd_livre', 'fwd', False, 'CG dianteiro, sem trimagem'),
         ('fwd_trim',  'fwd', True,  'CG dianteiro, com trimagem'),
         ('aft_livre', 'aft', False, 'CG traseiro, sem trimagem'),
         ('aft_trim',  'aft', True,  'CG traseiro, com trimagem')]

# criterio -> (coluna do fs, plano do cl_max, fator sobre o Re local)
# fator None significa cos(Lambda)
CRITERIOS = {'roteiro':  ('cl_norm', 'corrente', 1.0),
             'normal':   ('cl_norm', 'normal',   None),
             'corrente': ('cl',      'corrente', 1.0)}
CRIT_OFICIAL = 'roteiro'

# Varredura para localizar o estol. Atencao: com o cartao ANGLE da asa, o alpha
# do AVL e o angulo da FUSELAGEM; a asa ve alpha + i_w. O teto precisa cobrir
# tambem o criterio 'corrente', que exige cl (e nao cl_norm) chegar ao cl_max do
# perfil e por isso so estola bem mais tarde.
ALPHAS = np.arange(0.0, 18.01, 0.5)


# ------------------------------------------------------------------ utilidades

def razao_max(res, clmax_fn, Re_por_metro, coluna, fator_Re):
    '''max(coluna/cl_max) e a estacao critica de um resultado do AVL.'''
    w = at.wing_strips(res['fs'])
    lim = np.asarray(clmax_fn(Re_por_metro*fator_Re*w['Chord']), dtype=float)
    raz = w[coluna]/lim
    i = int(np.argmax(raz))
    return float(raz[i]), i, w, lim, raz


def alpha_do_estol(alphas, razoes):
    '''Alpha em que max(cl/cl_max) cruza 1, por interpolacao linear.'''
    r = np.asarray(razoes, dtype=float)
    if r[0] >= 1.0:
        raise RuntimeError('ja estolado em alpha=%.2f (razao=%.4f)'
                           % (alphas[0], r[0]))
    if r[-1] <= 1.0:
        raise RuntimeError('sem estol ate alpha=%.2f (razao=%.4f)'
                           % (alphas[-1], r[-1]))
    k = int(np.argmax(r >= 1.0))
    a0, a1 = alphas[k-1], alphas[k]
    r0, r1 = r[k-1], r[k]
    return float(a0 + (1.0 - r0)*(a1 - a0)/(r1 - r0))


# ------------------------------------------------------------------ calculo

def calcula():
    '''Roda o AVL e devolve o dicionario completo de resultados.'''
    ap = oc.run_designTool(oc.get_baseline())
    I, G = ap['inputs'], ap['geometry']
    pp = pj.ponto_projeto(ap)
    bv = pj.condicao_baixa_velocidade(ap)
    cosL = float(np.cos(I['sweep_w']))
    its = pj.incidencias_eh()        # item 3 (resultados_q3/)
    iw = pj.angulo_asa()

    clmax = {}
    for plano in ('corrente', 'normal'):
        mach = bv['M'] if plano == 'corrente' else bv['M']*cosL
        clmax[plano] = xp.carrega(mach=mach, plano=plano)

    b2 = 0.5*pp['bref']
    print('Baixa velocidade: V2=%.2f m/s  M=%.4f  Re/m=%.4e  cos(Lambda)=%.4f'
          % (bv['V2'], bv['M'], bv['Re_por_metro'], cosL))
    for plano in ('corrente', 'normal'):
        fn = clmax[plano][1]
        f = 1.0 if plano == 'corrente' else cosL
        print('  cl_max (%-8s): %.4f na raiz -> %.4f na ponta'
              % (plano, fn(bv['Re_por_metro']*f*G['cr_w'])[0],
                 fn(bv['Re_por_metro']*f*G['ct_w'])[0]))
    print('')

    res = {}
    for nome, cg, trim, rotulo in CASOS:
        it = its[cg]['it_deg']
        arq = its[cg]['arquivo']
        varre = at.run_cases(
            arq, bv['M'], [{'alpha': float(a), 'trim': trim} for a in ALPHAS],
            it=it, tag='sw_' + nome)
        crit = {}
        for cnome, (coluna, plano, fator) in CRITERIOS.items():
            fn = clmax[plano][1]
            f = cosL if fator is None else fator
            razoes = [razao_max(r, fn, bv['Re_por_metro'], coluna, f)[0]
                      for r in varre]
            a_est = alpha_do_estol(ALPHAS, razoes)
            r = at.run_cases(arq, bv['M'], [{'alpha': a_est, 'trim': trim}],
                             it=it, tag='st_%s_%s' % (nome, cnome))[0]
            rz, i, w, lim, raz = razao_max(r, fn, bv['Re_por_metro'], coluna, f)
            crit[cnome] = {
                'coluna': coluna, 'plano_clmax': plano,
                'alpha_max_deg': float(a_est), 'CLmax': float(r['CLtot']),
                'CD': float(r['CDtot']), 'Cm': float(r['Cmtot']),
                'delta_e_deg': float(r['elevator']),
                'y_estol_m': float(w['Yle'][i]),
                'eta_estol': float(w['Yle'][i]/b2),
                'cl_secao_estol': float(w[coluna][i]),
                'clmax_local': float(lim[i]),
                'razao_final': float(rz),
                'varredura': {'alpha': ALPHAS.tolist(),
                              'razao_max': [float(v) for v in razoes]},
                'dist': {'y': w['Yle'].tolist(), 'chord': w['Chord'].tolist(),
                         'cl_norm': w['cl_norm'].tolist(),
                         'cl': w['cl'].tolist(),
                         'clmax_lim': lim.tolist(),
                         'razao': raz.tolist()}}
            print('%-10s %-9s alpha_max=%6.3f  CLmax=%6.4f  delta_e=%+7.3f  '
                  'estol em eta=%.3f (%s=%.4f / cl_max=%.4f)'
                  % (nome, cnome, a_est, r['CLtot'], r['elevator'],
                     w['Yle'][i]/b2, coluna, w[coluna][i], lim[i]))
        res[nome] = dict(crit[CRIT_OFICIAL])
        res[nome].update({'rotulo': rotulo, 'cg': cg, 'arquivo': arq,
                          'trim': trim, 'it_deg': it, 'mach': bv['M'],
                          'Xref': float(varre[0]['Xref']),
                          'criterios': crit,
                          'n_aval_avl': len(varre) + len(CRITERIOS)})

    return {'ponto_projeto': pp, 'baixa_velocidade': bv,
            'cos_sweep': cosL, 'criterio_oficial': CRIT_OFICIAL,
            'clmax_perfil': {k: v[0] for k, v in clmax.items()},
            'casos': res,
            'geometria': {k: float(v) for k, v in G.items()
                          if isinstance(v, (int, float))},
            'iw_deg': float(iw),
            'controles': {k: float(I[k]) for k in
                          ('b_flap_b_wing', 'b_slat_b_wing', 'b_ail_b_wing',
                           'taper_w', 'tcr_w', 'tct_w', 'clmax_w')},
            'CLmax_clean_DT': float(0.9*I['clmax_w']*cosL)}


# ------------------------------------------------------------------ saidas

def escreve_saidas(dados):
    pp = dados['ponto_projeto']
    bv = dados['baixa_velocidade']
    res = dados['casos']
    geo = dados['geometria']
    ctl = dados['controles']
    cosL = dados['cos_sweep']
    dcl = dados['clmax_perfil']
    b2 = 0.5*pp['bref']

    with open(os.path.join(OUT, 'q4_secao_critica.json'), 'w',
              encoding='utf-8') as f:
        json.dump(dados, f, indent=2)

    # ---- tabela principal (criterio oficial)
    with open(os.path.join(OUT, 'tabela_secao_critica.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['caso', 'CG', 'trimado', 'it_deg', 'alpha_max_deg',
                     'CLmax', 'delta_e_deg', 'y_estol_m', 'eta_estol',
                     'cl_norm_estol', 'clmax_local'])
        for nome, _, _, _ in CASOS:
            c = res[nome]
            wr.writerow([nome, c['cg'], c['trim'], '%.2f' % c['it_deg'],
                         '%.3f' % c['alpha_max_deg'], '%.4f' % c['CLmax'],
                         '%.3f' % c['delta_e_deg'], '%.3f' % c['y_estol_m'],
                         '%.4f' % c['eta_estol'],
                         '%.4f' % c['cl_secao_estol'],
                         '%.4f' % c['clmax_local']])

    # ---- sensibilidade ao criterio
    with open(os.path.join(OUT, 'tabela_criterios.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['caso', 'criterio', 'coluna_fs', 'plano_clmax',
                     'alpha_max_deg', 'CLmax', 'delta_e_deg', 'eta_estol'])
        for nome, _, _, _ in CASOS:
            for cn in ('roteiro', 'normal', 'corrente'):
                c = res[nome]['criterios'][cn]
                wr.writerow([nome, cn, c['coluna'], c['plano_clmax'],
                             '%.3f' % c['alpha_max_deg'], '%.4f' % c['CLmax'],
                             '%.3f' % c['delta_e_deg'],
                             '%.4f' % c['eta_estol']])

    # ---- distribuicoes
    for nome, _, _, _ in CASOS:
        d = res[nome]['dist']
        with open(os.path.join(OUT, 'dist_%s.csv' % nome), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['y_m', 'eta', 'chord_m', 'cl_norm', 'cl',
                         'clmax_limite', 'cl_norm_sobre_clmax'])
            for k in range(len(d['y'])):
                wr.writerow(['%.4f' % d['y'][k], '%.4f' % (d['y'][k]/b2),
                             '%.4f' % d['chord'][k], '%.5f' % d['cl_norm'][k],
                             '%.5f' % d['cl'][k], '%.5f' % d['clmax_lim'][k],
                             '%.5f' % d['razao'][k]])

    # ---- tabelas LaTeX
    with open(os.path.join(TEX, 'tab_secao_critica.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lccccc}\n\\toprule\n')
        f.write('Caso & $i_t$ [$^\\circ$] & $\\alpha_{max}$ [$^\\circ$] & '
                '$C_{L\\,max}$ & $\\delta_e$ [$^\\circ$] & '
                '$\\eta$ do estol \\\\\n\\midrule\n')
        for nome, _, _, rotulo in CASOS:
            c = res[nome]
            f.write('%s & $%+.3f$ & %.2f & %.3f & $%+.2f$ & %.3f \\\\\n'
                    % (rotulo, c['it_deg'], c['alpha_max_deg'], c['CLmax'],
                       c['delta_e_deg'], c['eta_estol']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    rot_crit = {
        'roteiro': 'roteiro: $c_\\ell^{\\,norm}$ vs.\\ perfil da corrente',
        'normal': 'coerente: $c_\\ell^{\\,norm}$ vs.\\ perfil do plano normal',
        'corrente': 'sem correcao: $c_\\ell$ vs.\\ perfil da corrente'}
    with open(os.path.join(TEX, 'tab_criterios.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lcccc}\n\\toprule\n')
        f.write('Criterio de estol & \\multicolumn{2}{c}{CG dianteiro} & '
                '\\multicolumn{2}{c}{CG traseiro} \\\\\n')
        f.write('& $\\alpha_{max}$ [$^\\circ$] & $C_{L\\,max}$ '
                '& $\\alpha_{max}$ [$^\\circ$] & $C_{L\\,max}$ '
                '\\\\\n\\midrule\n')
        for cn in ('roteiro', 'normal', 'corrente'):
            cf = res['fwd_trim']['criterios'][cn]
            ca = res['aft_trim']['criterios'][cn]
            f.write('%s & %.2f & %.3f & %.2f & %.3f \\\\\n'
                    % (rot_crit[cn], cf['alpha_max_deg'], cf['CLmax'],
                       ca['alpha_max_deg'], ca['CLmax']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(os.path.join(TEX, 'tab_clmax_perfil.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{rrrrr}\n\\toprule\n')
        f.write('$c$ [m] & $\\eta$ & $Re$ [$\\times 10^{7}$] & '
                '$c_{\\ell\\,max}$ & $\\alpha_{c_{\\ell max}}$ [$^\\circ$] '
                '\\\\\n\\midrule\n')
        for c in sorted(dcl['corrente']['casos'], key=lambda c: -c['Re']):
            corda = c['Re']/bv['Re_por_metro']
            eta = (geo['cr_w'] - corda)/(geo['cr_w'] - geo['ct_w'])
            f.write('%.2f & %.2f & %.2f & %.3f & %.2f \\\\\n'
                    % (corda, min(max(eta, 0.0), 1.0), c['Re']/1e7,
                       c['clmax'], c['alpha_clmax']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    # ---- figuras
    cores = {'fwd_livre': 'tab:blue', 'fwd_trim': 'tab:cyan',
             'aft_livre': 'tab:red', 'aft_trim': 'tab:orange'}

    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, cg in zip(axs, ('fwd', 'aft')):
        d0 = res[cg + '_livre']['dist']
        ax.plot(np.array(d0['y'])/b2, d0['clmax_lim'], 'k--', lw=1.9,
                label=r'limite $c_{\ell\,max}(Re_{local})$ do perfil')
        # as duas curvas quase coincidem: a nao trimada vai grossa por baixo e
        # a trimada tracejada por cima, para que as duas fiquem visiveis
        for nome, lw, ls in ((cg + '_livre', 3.4, '-'),
                             (cg + '_trim', 1.8, '--')):
            c = res[nome]
            d = c['dist']
            ax.plot(np.array(d['y'])/b2, d['cl_norm'], color=cores[nome],
                    lw=lw, ls=ls,
                    label=(r'%s: $\alpha_{max}=%.2f^\circ$, $C_{L\,max}=%.3f$'
                           % ('sem trimagem' if not c['trim']
                              else 'com trimagem',
                              c['alpha_max_deg'], c['CLmax'])))
            ax.plot(c['eta_estol'], c['cl_secao_estol'], 'o',
                    color=cores[nome], ms=9, mfc='white', mew=1.9)
        ax.axvspan(1 - ctl['b_ail_b_wing'], 1.0, color='0.9', zorder=0)
        ax.annotate('faixa do aileron', (1 - ctl['b_ail_b_wing'] + 0.01, 0.12),
                    fontsize=8, color='0.35')
        ax.set_xlabel(r'$\eta = y/(b/2)$')
        ax.set_title('CG %s ($x_{cg}=%.2f$ m, $i_t=%+.3f^\\circ$)'
                     % ('dianteiro' if cg == 'fwd' else 'traseiro',
                        res[cg + '_livre']['Xref'],
                        res[cg + '_livre']['it_deg']), fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.95)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
    axs[0].set_ylabel(r'$c_\ell$ da secao no plano normal '
                      r'(coluna cl$\_$norm do AVL)')
    fig.suptitle('Metodo da secao critica -- asa limpa, $M=%.3f$, '
                 '$V_2=%.1f$ m/s ao nivel do mar' % (bv['M'], bv['V2']),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'cl_y_estol.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for nome, _, trim, rotulo in CASOS:
        d = res[nome]['dist']
        ax.plot(np.array(d['y'])/b2, d['razao'], color=cores[nome],
                lw=1.8 if trim else 3.4, ls='--' if trim else '-',
                label='%s ($\\alpha_{max}=%.2f^\\circ$)'
                      % (rotulo, res[nome]['alpha_max_deg']))
    ax.axhline(1.0, color='k', ls='--', lw=1.4)
    ax.axvspan(1 - ctl['b_ail_b_wing'], 1.0, color='0.9', zorder=0)
    ax.annotate('faixa do aileron', (1 - ctl['b_ail_b_wing'] + 0.01, 0.42),
                fontsize=8, color='0.35')
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$\eta = y/(b/2)$')
    ax.set_ylabel(r'$c_\ell / c_{\ell\,max}(Re_{local})$')
    ax.set_title('Grau de carregamento da secao na condicao de estol')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'razao_cl_clmax.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for plano, cor, lbl in (
            ('corrente', 'tab:blue', 'perfil da corrente livre'),
            ('normal', 'tab:green',
             r'perfil do plano normal ($t/c \div \cos\Lambda$)')):
        cs = sorted(dcl[plano]['casos'], key=lambda c: c['Re'])
        ax.plot([c['Re']/1e7 for c in cs], [c['clmax'] for c in cs], 'o-',
                color=cor, label=lbl)
    for lbl, c in (('ponta', geo['ct_w']), ('MAC', geo['cm_w']),
                   ('raiz', geo['cr_w'])):
        ax.axvline(bv['Re_por_metro']*c/1e7, color='0.75', lw=0.8, ls=':')
        ax.annotate(lbl, (bv['Re_por_metro']*c/1e7, ax.get_ylim()[0]),
                    fontsize=8, rotation=90, xytext=(3, 8),
                    textcoords='offset points')
    ax.set_xlabel(r'$Re$ [$\times 10^{7}$]')
    ax.set_ylabel(r'$c_{\ell\,max}$ (XFOIL)')
    ax.set_title(r'$c_{\ell\,max}$ do perfil do Lab 03 em funcao do Reynolds')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'clmax_perfil_Re.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for cn, cor in (('roteiro', 'tab:blue'), ('normal', 'tab:green'),
                    ('corrente', 'tab:red')):
        c = res['aft_trim']['criterios'][cn]
        v = c['varredura']
        ax.plot(v['alpha'], v['razao_max'], 'o-', ms=3.5, color=cor,
                label='%s ($\\alpha_{max}=%.2f^\\circ$, $C_{L\\,max}=%.3f$)'
                      % (cn, c['alpha_max_deg'], c['CLmax']))
        ax.axvline(c['alpha_max_deg'], color=cor, lw=0.8, ls=':')
    ax.axhline(1.0, color='k', ls='--', lw=1.3)
    ax.set_xlabel(r'$\alpha$ [graus]')
    ax.set_ylabel(r'$\max_y\,(c_\ell / c_{\ell\,max})$')
    ax.set_title('Deteccao do estol -- CG traseiro com trimagem')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'deteccao_estol.png'), dpi=200)
    plt.close(fig)

    # ---- macros LaTeX
    interp = {}
    for plano in ('corrente', 'normal'):
        cs = sorted(dcl[plano]['casos'], key=lambda c: c['Re'])
        interp[plano] = (np.log10([c['Re'] for c in cs]),
                         np.array([c['clmax'] for c in cs]))

    def cl_de(plano, Re):
        lR, cm = interp[plano]
        return float(np.interp(np.log10(Re), lR, cm))

    Rm = bv['Re_por_metro']
    ct = res['aft_trim']
    i_crit = int(np.argmax(ct['dist']['razao']))
    # razao medida entre as colunas cl_norm e cl do comando fs, para confrontar
    # com 1/cos^2(Lambda) no texto
    razao_col = np.array(ct['dist']['cl_norm'])/np.array(ct['dist']['cl'])
    # faixa em que a secao esta a menos de 3% do estol (planalto de carga)
    eta_ct = np.array(ct['dist']['y'])/b2
    plat = eta_ct[np.array(ct['dist']['razao']) > 0.97]
    mac = {'scMach': '%.3f' % bv['M'], 'scVdois': '%.1f' % bv['V2'],
           'scRemac': '%.2f' % (bv['Re_cref']/1e7),
           'scCosL': '%.4f' % cosL,
           'scSweep': '%.1f' % np.degrees(np.arccos(cosL)),
           'scRazaoColunas': '%.4f' % np.median(razao_col),
           'scRazaoColunasMin': '%.4f' % razao_col.min(),
           'scRazaoColunasMax': '%.4f' % razao_col.max(),
           'scInvCosDois': '%.4f' % (1.0/cosL**2),
           'scClmaxRaiz': '%.3f' % cl_de('corrente', Rm*geo['cr_w']),
           'scClmaxPonta': '%.3f' % cl_de('corrente', Rm*geo['ct_w']),
           'scClmaxRaizN': '%.3f' % cl_de('normal', Rm*cosL*geo['cr_w']),
           'scClmaxPontaN': '%.3f' % cl_de('normal', Rm*cosL*geo['ct_w']),
           'scClmaxDT': '%.3f' % ctl['clmax_w'],
           'scCLmaxDT': '%.3f' % dados['CLmax_clean_DT'],
           'scRazaoDT': '%.0f' % (100*ct['CLmax']/dados['CLmax_clean_DT']),
           'scFatorCarga': '%.2f' % (np.array(ct['dist']['cl'])[i_crit]
                                     / ct['CLmax']),
           'scEtaPlatIni': '%.2f' % plat.min(),
           'scEtaPlatFim': '%.2f' % plat.max(),
           'scIw': '%.1f' % dados['iw_deg'],
           'scAlphaAsa': '%.2f' % (res['aft_trim']['alpha_max_deg']
                                   + dados['iw_deg']),
           'scEtaAil': '%.2f' % (1 - ctl['b_ail_b_wing']),
           'scEtaSlat': '%.2f' % ctl['b_slat_b_wing'],
           'scEtaFlap': '%.2f' % ctl['b_flap_b_wing'],
           'scAfilamento': '%.2f' % ctl['taper_w'],
           'scTcRaiz': '%.3f' % ctl['tcr_w'],
           'scTcPonta': '%.3f' % ctl['tct_w'],
           'scCordaRaiz': '%.2f' % geo['cr_w'],
           'scCordaPonta': '%.2f' % geo['ct_w']}
    suf = {'fwd_livre': 'FwdLivre', 'fwd_trim': 'FwdTrim',
           'aft_livre': 'AftLivre', 'aft_trim': 'AftTrim'}
    for nome, s in suf.items():
        c = res[nome]
        mac['scAlpha' + s] = '%.2f' % c['alpha_max_deg']
        mac['scCLmax' + s] = '%.3f' % c['CLmax']
        mac['scDe' + s] = '%+.2f' % c['delta_e_deg']
        mac['scEta' + s] = '%.3f' % c['eta_estol']
        for cn, cs2 in (('normal', 'Norm'), ('corrente', 'Corr')):
            mac['scCLmax' + cs2 + s] = '%.3f' % c['criterios'][cn]['CLmax']
            mac['scAlpha' + cs2 + s] = '%.2f' % (
                c['criterios'][cn]['alpha_max_deg'])
    for cg, s in (('fwd', 'Fwd'), ('aft', 'Aft')):
        mac['scIt' + s] = '%+.3f' % res[cg + '_livre']['it_deg']
        mac['scXcg' + s] = '%.2f' % res[cg + '_livre']['Xref']
        mac['scPerdaCL' + s] = '%.1f' % (
            100*(1 - res[cg + '_trim']['CLmax']/res[cg + '_livre']['CLmax']))
    with open(os.path.join(TEX, 'macros.tex'), 'w', encoding='utf-8') as f:
        f.write('% gerado por q4_secao_critica.py\n')
        for k in sorted(mac):
            f.write('\\newcommand{\\%s}{%s}\n' % (k, mac[k]))

    print('\narquivos gravados em', OUT)
    print('tabelas LaTeX em', TEX)


def main(tex_only=False):
    for d in (OUT, TEX):
        os.makedirs(d, exist_ok=True)
    caminho = os.path.join(OUT, 'q4_secao_critica.json')
    if tex_only:
        with open(caminho, encoding='utf-8') as f:
            dados = json.load(f)
        print('reaproveitando', caminho)
    else:
        dados = calcula()
    escreve_saidas(dados)


if __name__ == '__main__':
    main(tex_only=('--tex' in sys.argv))
