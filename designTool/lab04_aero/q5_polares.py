'''
PRJ-23 Lab 04 - Item 2.5: polares de arrasto da NJ-0502 no AVL.

Quatro polares no Mach e na altitude do ponto de projeto, todas com a
incidencia de empenagem do item 2.3 e as superficies de alta sustentacao
neutras:

  (a) fwd_livre  CG dianteiro, sem deflexoes de superficie de controle
  (b) aft_livre  CG traseiro,  sem deflexoes de superficie de controle
  (c) fwd_trim   CG dianteiro, arfagem compensada (CM = 0)
  (d) aft_trim   CG traseiro,  arfagem compensada (CM = 0)

Cada polar e varrida pela restricao de CL do AVL (comando "a c CL"), de
CL = -0,5 (valor sugerido no enunciado) ate o CLmax obtido pelo metodo da secao
critica do item 2.4, que e especifico de cada caso. O ponto de projeto e
avaliado a parte e marcado nas curvas.

O arquivo de saida guarda tambem alpha e delta_e de cada ponto, que sao os
dados dos itens 2.6 (CL x alpha) e 2.7 (CL x delta_e), e o ajuste quadratico
CD = CD0 + CD_alpha*alpha + CD_alpha2*alpha^2 pedido na Tab. 7 do enunciado
(feito sobre a polar nao trimada de CG traseiro, o caso 5.b).

Uso (a partir desta pasta):
    python q5_polares.py           # roda o AVL e grava tudo
    python q5_polares.py --tex     # refaz so CSV/tabelas/figuras
Saidas: ../resultados_polares/ e ../tex_polares/
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
import opt_common as oc                    # noqa: E402

RESCOM = os.path.join(LAB, 'resultados_comum')
SECCRIT = os.path.join(LAB, 'resultados_q4')
OUT = os.path.join(LAB, 'resultados_q5')
TEX = os.path.join(LAB, 'tex_q5')

CASOS = [('fwd_livre', 'fwd', False, 'CG dianteiro, sem deflexoes'),
         ('aft_livre', 'aft', False, 'CG traseiro, sem deflexoes'),
         ('fwd_trim',  'fwd', True,  'CG dianteiro, compensada ($C_M=0$)'),
         ('aft_trim',  'aft', True,  'CG traseiro, compensada ($C_M=0$)')]

CL_MIN = -0.5
N_PONTOS = 36
CASO_AJUSTE = 'aft_livre'      # item 5.b do enunciado


def polar_avl(arq, mach, it, trim, CLs, tag):
    casos = [{'CL': float(cl), 'trim': trim} for cl in CLs]
    rs = at.run_cases(arq, mach, casos, it=it, tag=tag)
    return {'CL': np.array([r['CLtot'] for r in rs]),
            'CD': np.array([r['CDtot'] for r in rs]),
            'CDind': np.array([r['CDind'] for r in rs]),
            'CDvis': np.array([r['CDvis'] for r in rs]),
            'alpha': np.array([r['Alpha'] for r in rs]),
            'delta_e': np.array([r['elevator'] for r in rs]),
            'Cm': np.array([r['Cmtot'] for r in rs])}


def calcula():
    '''Roda o AVL e devolve o dicionario completo de resultados.'''
    ap = oc.run_designTool(oc.get_baseline())
    pp = pj.ponto_projeto(ap)
    its = pj.incidencias_eh()        # item 3 (resultados_q3/)
    with open(os.path.join(SECCRIT, 'q4_secao_critica.json'),
              encoding='utf-8') as f:
        sc = json.load(f)['casos']

    print('Polares em M=%.4f, h=%.0f m. CL do ponto de projeto = %.4f'
          % (pp['M'], pp['h_m'], pp['CL']))

    pol = {}
    for nome, cg, trim, rotulo in CASOS:
        it = its[cg]['it_deg']
        CLmax = sc[nome]['CLmax']
        CLs = np.linspace(CL_MIN, CLmax, N_PONTOS)
        # garante o ponto de projeto na grade
        CLs = np.unique(np.concatenate([CLs, [pp['CL']]]))
        d = polar_avl(its[cg]['arquivo'], pp['M'], it, trim, CLs, 'pol_' + nome)
        # ponto de projeto: indice do CL mais proximo (esta na grade)
        i = int(np.argmin(np.abs(d['CL'] - pp['CL'])))
        d.update({'rotulo': rotulo, 'cg': cg, 'trim': trim, 'it_deg': it,
                  'CLmax': CLmax, 'i_projeto': i,
                  'CD_projeto': float(d['CD'][i]),
                  'alpha_projeto': float(d['alpha'][i]),
                  'delta_e_projeto': float(d['delta_e'][i])})
        pol[nome] = d
        print('%-10s it=%+5.1f  CLmax=%.4f  CD(ponto)=%.5f  alpha=%.3f  '
              'delta_e=%+.3f' % (nome, it, CLmax, d['CD_projeto'],
                                 d['alpha_projeto'], d['delta_e_projeto']))

    # ---------------------------------------------- polar do designTool
    CL_dt = np.linspace(CL_MIN, max(p['CLmax'] for p in pol.values()), 120)
    CD_dt = pj.polar_designTool(ap, pp, CL_dt)
    CD_dt_projeto = float(pj.polar_designTool(ap, pp, [pp['CL']])[0])
    print('designTool: CD(ponto de projeto) = %.5f' % CD_dt_projeto)

    # ---------------------------------------------- ajuste CD(alpha) do item 5.b
    d = pol[CASO_AJUSTE]
    m = (d['CL'] >= -0.1) & (d['CL'] <= d['CLmax'])
    a_rad = np.radians(d['alpha'][m])
    coef = np.polyfit(a_rad, d['CD'][m], 2)          # [CD_a2, CD_a, CD0]
    ajuste = {'caso': CASO_AJUSTE, 'CD0': float(coef[2]),
              'CD_alpha_por_rad': float(coef[1]),
              'CD_alpha2_por_rad2': float(coef[0]),
              'faixa_CL': [float(d['CL'][m].min()), float(d['CL'][m].max())],
              'r2': float(1 - np.sum((d['CD'][m] - np.polyval(coef, a_rad))**2)
                          / np.sum((d['CD'][m] - d['CD'][m].mean())**2))}
    print('Ajuste CD = CD0 + CDa*alpha + CDa2*alpha^2 (caso %s): '
          'CD0=%.5f  CDa=%.5f/rad  CDa2=%.5f/rad^2  R2=%.5f'
          % (CASO_AJUSTE, ajuste['CD0'], ajuste['CD_alpha_por_rad'],
             ajuste['CD_alpha2_por_rad2'], ajuste['r2']))

    return {'ponto_projeto': pp, 'CD_designTool_no_ponto': CD_dt_projeto,
            'CD0_designTool': pp['CD0_cruzeiro'],
            'ajuste_quadratico': ajuste,
            'polar_designTool': {'CL': CL_dt.tolist(), 'CD': CD_dt.tolist()},
            'polares': {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray)
                                 else vv) for kk, vv in v.items()}
                        for k, v in pol.items()}}


def escreve_saidas(dados):
    '''Grava JSON, CSV, tabelas LaTeX e figuras a partir de calcula().'''
    pp = dados['ponto_projeto']
    CD_dt_projeto = dados['CD_designTool_no_ponto']
    ajuste = dados['ajuste_quadratico']
    CL_dt = np.array(dados['polar_designTool']['CL'])
    CD_dt = np.array(dados['polar_designTool']['CD'])
    pol = {k: {kk: (np.array(vv) if isinstance(vv, list) else vv)
               for kk, vv in v.items()} for k, v in dados['polares'].items()}

    with open(os.path.join(OUT, 'q5_polares.json'), 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2)

    for nome, _, _, _ in CASOS:
        d = pol[nome]
        with open(os.path.join(OUT, 'polar_%s.csv' % nome), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['CL', 'CD', 'CDind', 'CDvis', 'alpha_deg',
                         'delta_e_deg', 'Cm'])
            for k in range(len(d['CL'])):
                wr.writerow(['%.5f' % d['CL'][k], '%.6f' % d['CD'][k],
                             '%.6f' % d['CDind'][k], '%.6f' % d['CDvis'][k],
                             '%.4f' % d['alpha'][k], '%.4f' % d['delta_e'][k],
                             '%.6f' % d['Cm'][k]])

    with open(os.path.join(OUT, 'polar_designTool.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['CL', 'CD'])
        for cl, cd in zip(CL_dt, CD_dt):
            wr.writerow(['%.5f' % cl, '%.6f' % cd])

    with open(os.path.join(OUT, 'q5_tabela_CD_ponto_projeto.csv'), 'w',
              newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['configuracao', 'CD_AVL', 'alpha_deg', 'delta_e_deg',
                     'CD_designTool', 'dif_percentual'])
        for nome, _, _, rotulo in CASOS:
            d = pol[nome]
            wr.writerow([nome, '%.6f' % d['CD_projeto'],
                         '%.4f' % d['alpha_projeto'],
                         '%.4f' % d['delta_e_projeto'],
                         '%.6f' % CD_dt_projeto,
                         '%.2f' % (100*(d['CD_projeto']/CD_dt_projeto - 1))])

    # ---------------------------------------------- tabelas tex
    with open(os.path.join(TEX, 'tab_CD_ponto_projeto.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lccccc}\n\\toprule\n')
        f.write('Configuracao & $\\alpha$ [$^\\circ$] & $\\delta_e$ [$^\\circ$] '
                '& $C_D$ (AVL) & $C_D$ (designTool) & '
                'dif. \\\\\n\\midrule\n')
        for nome, _, _, rotulo in CASOS:
            d = pol[nome]
            de = d['delta_e_projeto']
            de = 0.0 if abs(de) < 0.005 else de      # evita imprimir "-0,00"
            f.write('%s & %.2f & $%+.2f$ & %.5f & %.5f & $%+.1f$\\%% \\\\\n'
                    % (rotulo, d['alpha_projeto'], de,
                       d['CD_projeto'], CD_dt_projeto,
                       100*(d['CD_projeto']/CD_dt_projeto - 1)))
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(os.path.join(TEX, 'tab_ajuste_CD.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{llr}\n\\toprule\n')
        f.write('Parametro & Explicacao & Valor \\\\\n\\midrule\n')
        f.write('$C_{D0}$ & termo constante ($\\alpha$ da fuselagem; '
                'nao e o arrasto parasita) & %.5f \\\\\n' % ajuste['CD0'])
        f.write('$C_{D\\alpha}$ & termo linear [1/rad] & %.5f \\\\\n'
                % ajuste['CD_alpha_por_rad'])
        f.write('$C_{D\\alpha^2}$ & termo quadratico [1/rad$^2$] & %.5f \\\\\n'
                % ajuste['CD_alpha2_por_rad2'])
        f.write('\\bottomrule\n\\end{tabular}\n')

    # ---------------------------------------------- figuras
    # o ultimo campo e o tamanho do marcador do ponto de projeto: as quatro
    # configuracoes praticamente coincidem ali, entao os marcadores sao
    # aninhados para que todos fiquem visiveis
    estilo = {'fwd_livre': ('tab:blue', '-', 13.0),
              'aft_livre': ('tab:red', '-', 10.5),
              'fwd_trim': ('tab:cyan', '--', 8.0),
              'aft_trim': ('tab:orange', '--', 5.5)}

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(CD_dt, CL_dt, color='0.45', lw=1.4, ls=':',
            label='polar do designTool')
    for nome, _, _, rotulo in CASOS:
        d = pol[nome]
        cor, ls, ms = estilo[nome]
        ax.plot(d['CD'], d['CL'], color=cor, ls=ls, lw=1.8, label=rotulo)
        i = d['i_projeto']
        ax.plot(d['CD'][i], d['CL'][i], 'o', color=cor, ms=ms, mfc='none',
                mew=1.8)
    ax.plot(CD_dt_projeto, pp['CL'], 's', color='0.25', ms=8, mfc='white',
            mew=1.8, label='ponto de projeto')
    ax.set_xlabel(r'$C_D$')
    ax.set_ylabel(r'$C_L$')
    ax.set_title('Polar de arrasto da NJ-0502 (AVL, $M=%.2f$, $h=%.0f$ m)'
                 % (pp['M'], pp['h_m']))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'polar_CD_CL.png'), dpi=200)
    plt.close(fig)

    # zoom na vizinhanca do ponto de projeto
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(CD_dt, CL_dt, color='0.45', lw=1.4, ls=':',
            label='polar do designTool')
    for nome, _, _, rotulo in CASOS:
        d = pol[nome]
        cor, ls, ms = estilo[nome]
        ax.plot(d['CD'], d['CL'], color=cor, ls=ls, lw=1.8, label=rotulo)
        i = d['i_projeto']
        ax.plot(d['CD'][i], d['CL'][i], 'o', color=cor, ms=ms, mfc='none',
                mew=1.8)
    ax.plot(CD_dt_projeto, pp['CL'], 's', color='0.25', ms=8, mfc='white',
            mew=1.8)
    ax.set_xlim(0.014, 0.032)
    ax.set_ylim(0.0, 0.85)
    ax.set_xlabel(r'$C_D$')
    ax.set_ylabel(r'$C_L$')
    ax.set_title('Detalhe em torno do ponto de projeto ($C_L=%.3f$)' % pp['CL'])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'polar_CD_CL_zoom.png'), dpi=200)
    plt.close(fig)

    # ajuste quadratico CD(alpha) do item 5.b
    d = pol[CASO_AJUSTE]
    m = (d['CL'] >= -0.1) & (d['CL'] <= d['CLmax'])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(d['alpha'][m], d['CD'][m], 'o', ms=5, label='AVL (%s)'
            % pol[CASO_AJUSTE]['rotulo'])
    aa = np.linspace(d['alpha'][m].min(), d['alpha'][m].max(), 200)
    ax.plot(aa, np.polyval(
        [ajuste['CD_alpha2_por_rad2'], ajuste['CD_alpha_por_rad'],
         ajuste['CD0']], np.radians(aa)), '-',
        label=r'$C_D = %.5f %+.5f\,\alpha %+.5f\,\alpha^2$'
              % (ajuste['CD0'], ajuste['CD_alpha_por_rad'],
                 ajuste['CD_alpha2_por_rad2']))
    ax.set_xlabel(r'$\alpha$ [graus]')
    ax.set_ylabel(r'$C_D$')
    ax.set_title(r'Ajuste quadratico da polar nao trimada ($R^2=%.5f$)'
                 % ajuste['r2'])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ajuste_CD_alpha.png'), dpi=200)
    plt.close(fig)

    # ---------------------------------------------- macros tex
    mac = {'poMach': '%.2f' % pp['M'], 'poAlt': '%.0f' % pp['h_m'],
           'poCLproj': '%.4f' % pp['CL'],
           'poCDdt': '%.5f' % CD_dt_projeto,
           'poCDzeroDT': '%.5f' % dados['CD0_designTool'],
           'poCDzeroFit': '%.5f' % ajuste['CD0'],
           'poCDaFit': '%.5f' % ajuste['CD_alpha_por_rad'],
           'poCDaaFit': '%.5f' % ajuste['CD_alpha2_por_rad2'],
           'poRdois': '%.5f' % ajuste['r2'],
           'poCLmin': '%.1f' % CL_MIN,
           'poIw': '%.1f' % pp['iw_deg'],
           'poEffDT': '%.1f' % (pp['CL']/CD_dt_projeto)}
    suf = {'fwd_livre': 'FwdLivre', 'aft_livre': 'AftLivre',
           'fwd_trim': 'FwdTrim', 'aft_trim': 'AftTrim'}
    for nome, s in suf.items():
        d = pol[nome]
        mac['poCD' + s] = '%.5f' % d['CD_projeto']
        mac['poDif' + s] = '%+.1f' % (100*(d['CD_projeto']/CD_dt_projeto - 1))
        mac['poDifAbs' + s] = '%.1f' % abs(
            100*(d['CD_projeto']/CD_dt_projeto - 1))
        mac['poAlpha' + s] = '%.2f' % d['alpha_projeto']
        mac['poDe' + s] = '%+.2f' % d['delta_e_projeto']
        mac['poCLmax' + s] = '%.3f' % d['CLmax']
        mac['poEff' + s] = '%.1f' % (pp['CL']/d['CD_projeto'])
    for cg, s in (('fwd', 'Fwd'), ('aft', 'Aft')):
        mac['poIt' + s] = '%+.3f' % pol[cg + '_livre']['it_deg']
    with open(os.path.join(TEX, 'macros.tex'), 'w', encoding='utf-8') as f:
        f.write('% gerado por q5_polares.py\n')
        for k in sorted(mac):
            f.write('\\newcommand{\\%s}{%s}\n' % (k, mac[k]))

    print('\narquivos gravados em', OUT)
    print('tabelas LaTeX em', TEX)


def main(tex_only=False):
    for d in (OUT, TEX):
        os.makedirs(d, exist_ok=True)
    caminho = os.path.join(OUT, 'q5_polares.json')
    if tex_only:
        with open(caminho, encoding='utf-8') as f:
            dados = json.load(f)
        print('reaproveitando', caminho)
    else:
        dados = calcula()
    escreve_saidas(dados)


if __name__ == '__main__':
    main(tex_only=('--tex' in sys.argv))
