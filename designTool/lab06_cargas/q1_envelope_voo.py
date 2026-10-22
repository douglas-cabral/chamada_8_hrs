'''
PRJ-23 Atividade 06 - Item 1: envelope de voo (diagramas V-n) da NJ-0502.

Duas condicoes de peso, ambas na altitude e no Mach de cruzeiro:
  - inicio do cruzeiro: W_ini (fim da subida, fuel_weight do designTool);
  - fim do cruzeiro:    W_fim = W_ini * Mf_cruise.
Sao os mesmos W_ini/W_fim do ponto de projeto do Lab 03/Lab 04
(lab03_perfil/airfoil_mod/run_1_ponto_projeto.py).

Metodologia (FAR/CS-25, tudo em velocidade equivalente EAS):
  - Manobra (25.333/25.337): n_max = 2,1 + 24000/(W[lb] + 10000), limitado
    a [2,5; 3,8]; n_min = -1 ate V_C e sobe linearmente ate 0 em V_D.
    Linhas de estol: n = rho0 V^2 S CL / (2 W), com CL_max e CL_min.
    V_S1 = sqrt(2W/(rho0 S CL_max)), V_A = V_S1 sqrt(n_max).
  - Velocidades de projeto (25.335), definidas pelo professor em sala:
    M_C = MMO = 0,89 e M_D = MMO + 0,07 = 0,96, na altitude de cruzeiro.
  - Rajada (25.341, formula de Pratt): U_ref = 56 ft/s EAS ao nivel do mar,
    44 ft/s a 15000 ft e 20,86 ft/s a 60000 ft (interpolacao linear);
    U_ref em V_B e V_C, U_ref/2 em V_D.
      mu_g = 2 (W/S) / (rho c_mg CL_alpha g),  K_g = 0,88 mu_g/(5,3 + mu_g)
      Delta_n = K_g rho0 U_de V_E CL_alpha / (2 W/S)
    V_B = V_S1 sqrt(1 + Delta_n(V_C, U_ref)) (25.335(d)), sem passar de V_C.
  - CL_alpha: aviao completo no AVL, em M_C (onde a rajada e avaliada) e,
    para registro, no Mach de cruzeiro. Modelo: (fwd.avl oficial, hoje com washout de
    -7 deg, e o i_t correspondente do Lab 04; controles travados) no Mach
    de cruzeiro.

Uso: python q1_envelope_voo.py         (roda designTool + AVL)
     python q1_envelope_voo.py --tex   (refaz figuras/tabelas do JSON gravado)
     python q1_envelope_voo.py --clmax 1.3   (variante com outro CL_max,
     python q1_envelope_voo.py --clmax avl    em resultados_q1/clmax_<tag>/)
'''

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNTOOL_DIR = os.path.abspath(os.path.join(HERE, '..'))
LAB04 = os.path.join(DESIGNTOOL_DIR, 'lab04_aero')
RESDIR = os.path.join(HERE, 'resultados_q1')
TEXDIR = os.path.join(HERE, 'tex_q1')
JSON_OUT = os.path.join(RESDIR, 'q1_envelope.json')
Q4_JSON = os.path.join(LAB04, 'resultados_q4', 'q4_secao_critica.json')
WASHOUT_CSV = os.path.join(LAB04, 'resultados_washout', 'tabela_washout.csv')
for p in (DESIGNTOOL_DIR, LAB04):
    if p not in sys.path:
        sys.path.insert(0, p)

from designTool.auxiliary import atmosphere                # noqa: E402
from designTool.constants import ft2m, gravity, lb2N       # noqa: E402

RHO0 = atmosphere(0.0)['density']
KT = 0.514444                    # m/s por no

# Fonte do CL_max de asa limpa:
#   'designTool' -> CLmax_clean = 0,9 cl_max cos(Lambda) (1,304)
#   'avl_q4'     -> metodo da secao critica do Lab 04, CG dianteiro trimado
#                   (0,854). Com ele a linha de estol do inicio do cruzeiro
#                   nao alcanca n_max antes de V_D e o ponto A nao existe.
CLMAX_FONTE = 'designTool'
CLMIN = -0.8                     # CL_min fixo (definido pelo professor)
MMO = 0.89                       # M_C = MMO (definido pelo professor)
DELTA_MD = 0.07                  # M_D = MMO + 0,07 (idem)


# ------------------------------------------------------------------ normas

def u_ref(h):
    '''Velocidade de rajada de referencia, FAR 25.341(a)(5)(i) [m/s EAS].'''
    h_ft = h/ft2m
    u_ft = np.interp(h_ft, [0.0, 15000.0, 60000.0], [56.0, 44.0, 20.86])
    return float(u_ft*ft2m)


def n_max_far25(W):
    '''Fator de carga limite positivo, FAR 25.337(b).'''
    W_lb = W/lb2N
    return float(np.clip(2.1 + 24000.0/(W_lb + 10000.0), 2.5, 3.8))


def kg_pratt(WS, rho, c_mg, CLa):
    mu = 2.0*WS/(rho*c_mg*CLa*gravity)
    return mu, 0.88*mu/(5.3 + mu)


def delta_n_rajada(VE, Ude, WS, Kg, CLa):
    return Kg*RHO0*Ude*VE*CLa/(2.0*WS)


# ------------------------------------------------------------------ calculo

def cl_alpha_avl(M, it):
    '''CL_alpha do aviao completo [1/rad], controles travados.'''
    import avl_tools as at
    alphas = [0.0, 2.0]
    r = at.run_cases('fwd.avl', M, [{'alpha': a} for a in alphas], it=it,
                     tag='q1_cla')
    CL = [x['CLtot'] for x in r]
    return float((CL[1] - CL[0])/np.radians(alphas[1] - alphas[0])), CL


def washout_fwd_avl():
    '''Torcao da ultima secao da Wing no fwd.avl oficial [deg].'''
    caminho = os.path.join(DESIGNTOOL_DIR, 'AVL_package', 'fwd.avl')
    linhas = open(caminho, encoding='utf-8',
                  errors='replace').read().splitlines()
    surf, eps = None, 0.0
    for k, ln in enumerate(linhas):
        s = ln.strip().upper()
        if s == 'SURFACE':
            surf = linhas[k + 1].strip()
        elif s == 'SECTION' and surf == 'Wing':
            eps = float(linhas[k + 1].split()[4])
    return eps


def it_e_clmax_avl(pj):
    '''i_t do CG dianteiro e CL_max da secao critica para a asa do fwd.avl.

    Asa reta: item 3 / item 4 do Lab 04. Asa torcida (fwd.avl oficial desde
    b8f7de4): linha correspondente de resultados_washout/tabela_washout.csv,
    que refaz o i_t e a secao critica para cada torcao.
    '''
    eps = washout_fwd_avl()
    if abs(eps) < 1e-6:
        it = pj.incidencias_eh()['fwd']['it_deg']
        q4 = json.load(open(Q4_JSON, encoding='utf-8'))
        return eps, it, float(q4['casos']['fwd_trim']['CLmax'])
    with open(WASHOUT_CSV, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if abs(float(r['washout_deg']) - eps) < 1e-3:
                return eps, float(r['it_fwd_deg']), float(r['CLmax_fwd_trim'])
    raise RuntimeError('torcao %.3f deg do fwd.avl nao esta em %s'
                       % (eps, WASHOUT_CSV))


def clmax_fonte(pp, clmax_avl):
    if CLMAX_FONTE == 'designTool':
        return pp['CLmax_clean_DT'], 'designTool: 0,9 cl_max cos(Lambda)'
    return clmax_avl, 'AVL, secao critica (Lab 04, CG dianteiro trimado)'


def envelope(W, base):
    '''Pontos do diagrama V-n para um peso W [N].'''
    S, h = base['Sref'], base['h_m']
    rho, a = base['rho'], base['a']
    CLmax, CLmin, CLa = base['CLmax'], base['CLmin'], base['CLa']
    c_mg = base['c_mg']
    sq = np.sqrt(rho/RHO0)
    WS = W/S

    n_pos = n_max_far25(W)
    n_neg = -1.0
    VS1 = np.sqrt(2*WS/(RHO0*CLmax))
    VSn = np.sqrt(2*WS/(RHO0*abs(CLmin)))     # n = -1 sobre o estol negativo
    VA = VS1*np.sqrt(n_pos)
    VC = base['MC']*a*sq
    VD = base['MD']*a*sq

    U = u_ref(h)
    mu, Kg = kg_pratt(WS, rho, c_mg, CLa)
    dnC = delta_n_rajada(VC, U, WS, Kg, CLa)
    VB = min(VS1*np.sqrt(1.0 + dnC), VC)
    dnB = delta_n_rajada(VB, U, WS, Kg, CLa)
    dnD = delta_n_rajada(VD, 0.5*U, WS, Kg, CLa)

    def tas(VE):
        return VE/sq

    vels = {'VS1': VS1, 'VS_neg': VSn, 'VA': VA, 'VB': VB, 'VC': VC,
            'VD': VD}
    return {
        'W_N': W, 'W_kgf': W/gravity, 'WS': WS,
        'n_max': n_pos, 'n_min': n_neg,
        'mu_g': mu, 'K_g': Kg, 'U_ref': U,
        'dn_B': dnB, 'dn_C': dnC, 'dn_D': dnD,
        'n_stall_VC': RHO0*VC**2*CLmax/(2*WS),
        'velocidades_EAS': vels,
        'velocidades_TAS': {k: tas(v) for k, v in vels.items()},
        'mach': {k: tas(v)/a for k, v in vels.items()},
        'VA_maior_que_VC': bool(VA > VC),
        'VA_existe': bool(VA <= VD),
    }


def calcula():
    import ponto_projeto as pj
    pp = pj.ponto_projeto()
    eps, it, clmax_avl = it_e_clmax_avl(pj)
    W0 = pp['W0_kgf']*gravity
    W_ini = pp['W_ini_kgf']*gravity
    W_fim = pp['W_fim_kgf']*gravity

    CLa, CL_avl = cl_alpha_avl(MMO, it)
    CLa_cruz, CL_avl_cruz = cl_alpha_avl(pp['M'], it)
    CLmax, fonte = clmax_fonte(pp, clmax_avl)
    base = {
        'Sref': pp['Sref'], 'cref': pp['cref'], 'bref': pp['bref'],
        'c_mg': pp['Sref']/pp['bref'],
        'h_m': pp['h_m'], 'h_ft': pp['h_ft'],
        'rho': pp['rho'], 'a': pp['a'], 'sigma': pp['rho']/RHO0,
        'M_cruzeiro': pp['M'], 'MC': MMO, 'MD': MMO + DELTA_MD,
        'CLmax': CLmax, 'CLmax_fonte': fonte,
        'CLmin': CLMIN,
        'CLa': CLa, 'CLa_deg': CLa*np.pi/180, 'CL_avl_alpha_0_2': CL_avl,
        'CLa_cruzeiro': CLa_cruz, 'CL_avl_alpha_0_2_cruzeiro': CL_avl_cruz,
        'it_deg': it, 'iw_deg': pp['iw_deg'], 'washout_deg': eps,
        'CLmax_DT': pp['CLmax_clean_DT'], 'CLmax_avl_q4': clmax_avl,
        'U_ref': u_ref(pp['h_m']),
        'W0_N': W0, 'W0_kgf': pp['W0_kgf'],
    }

    return {'base': base,
            'condicoes': {'inicio': envelope(W_ini, base),
                          'fim': envelope(W_fim, base)}}


# ------------------------------------------------------------------ saidas

ROTULO = {'inicio': 'Início do cruzeiro', 'fim': 'Fim do cruzeiro'}


def curvas(env, base):
    '''Contornos de manobra e de rajada para o grafico.'''
    v = env['velocidades_EAS']
    WS = env['WS']
    VS1, VSn, VA, VB = v['VS1'], v['VS_neg'], v['VA'], v['VB']
    VC, VD = v['VC'], v['VD']
    n_pos, n_neg = env['n_max'], env['n_min']

    # linha de estol positiva de n = 1 ate n_max (ou ate V_D, se V_A > V_D)
    V_top = min(VA, VD)
    Vp = np.linspace(VS1, V_top, 80)
    np_ = RHO0*Vp**2*base['CLmax']/(2*WS)
    Vn = np.linspace(0.0, VSn, 80)
    nn = -RHO0*Vn**2*abs(base['CLmin'])/(2*WS)
    # contorno de manobra: estol positivo -> n_max ate V_D -> V_D -> n_min
    man_x = list(Vp) + ([VD] if VA <= VD else []) + [VD, VC, VSn]
    man_y = list(np_) + ([n_pos] if VA <= VD else []) + [0.0, n_neg, n_neg]
    raj_x = [0.0, VB, VC, VD, VD, VC, VB, 0.0]
    raj_y = [1.0, 1 + env['dn_B'], 1 + env['dn_C'], 1 + env['dn_D'],
             1 - env['dn_D'], 1 - env['dn_C'], 1 - env['dn_B'], 1.0]
    return {'estol_pos': (Vp, np_), 'estol_neg': (Vn, nn),
            'manobra': (np.array(man_x), np.array(man_y)),
            'rajada': (np.array(raj_x), np.array(raj_y))}


def figura(res, chave, arquivo):
    base, env = res['base'], res['condicoes'][chave]
    c = curvas(env, base)
    v = env['velocidades_EAS']

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    # linhas de estol estendidas (pontilhadas) para leitura
    Vext = np.linspace(0, v['VD']*1.02, 200)
    ax.plot(Vext, RHO0*Vext**2*base['CLmax']/(2*env['WS']), ':', color='0.6',
            lw=0.8)
    ax.plot(Vext, -RHO0*Vext**2*abs(base['CLmin'])/(2*env['WS']), ':',
            color='0.6', lw=0.8)

    mx, my = c['manobra']
    ax.plot(mx, my, '-', color='C0', lw=2.0, label='Manobra')
    ax.plot(c['estol_neg'][0], c['estol_neg'][1], '-', color='C0', lw=2.0)
    rx, ry = c['rajada']
    ax.plot(rx, ry, '--', color='C3', lw=1.6, label='Rajada')
    # retas de rajada saindo de (0, 1)
    for Vk, dn in ((v['VB'], env['dn_B']), (v['VC'], env['dn_C']),
                   (v['VD'], env['dn_D'])):
        for s in (+1, -1):
            ax.plot([0, Vk], [1, 1 + s*dn], '-', color='C3', lw=0.5,
                    alpha=0.5)

    nmax = env['n_max']
    ylim = (min(-1.6, 1 - env['dn_C'] - 0.3),
            max(nmax, 1 + env['dn_C'], 1 + env['dn_B']) + 0.6)
    nomes = sorted((k for k in ('VS1', 'VA', 'VB', 'VC', 'VD')
                    if v[k] <= v['VD']*1.001), key=lambda k: v[k])
    for i, nome in enumerate(nomes):
        Vk = v[nome]
        ax.axvline(Vk, color='0.85', lw=0.8, zorder=0)
        # rotulos alternados em duas alturas para nao encavalar V_C/V_A/V_D
        ax.text(Vk, ylim[0] + 0.08 + 0.2*(i % 2), '$V_{%s}$' % nome[1:],
                ha='center',
                va='bottom', fontsize=9,
                bbox=dict(fc='white', ec='none', pad=0.5))
    ax.axhline(0, color='k', lw=0.6)
    ax.axhline(1, color='0.7', lw=0.6, ls='--')

    ax.set_xlim(0, v['VD']*1.08)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Velocidade equivalente $V_E$ [m/s]')
    ax.set_ylabel('Fator de carga $n$')
    ax.set_title('%s: W = %.0f kgf, h = %.0f ft, $M_C$ = %.2f, $M_D$ = %.2f'
                 % (ROTULO[chave], env['W_kgf'], base['h_ft'], base['MC'],
                    base['MD']))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', title='$C_{L_{max}}$ = %.3f' % base['CLmax'],
              title_fontsize=9)
    if not env['VA_existe']:
        ax.text(0.98, 0.97, '$V_A$ = %.1f m/s > $V_D$: estol a n = %.2f em $V_D$'
                % (v['VA'], RHO0*v['VD']**2*base['CLmax']/(2*env['WS'])),
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(fc='#fff3cd', ec='0.6'))

    # eixo superior em Mach na altitude de cruzeiro
    sq = np.sqrt(base['sigma'])
    sec = ax.secondary_xaxis('top', functions=(
        lambda ve: ve/sq/base['a'], lambda m: m*base['a']*sq))
    sec.set_xlabel('Mach em %.0f ft' % base['h_ft'])
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


def br(x, nd):
    return ('%.*f' % (nd, x)).replace('.', '{,}')


def milhar(x, nd=0):
    s = '%.*f' % (nd, x)
    inteiro, _, frac = s.partition('.')
    g = []
    while inteiro:
        g.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    return r'\,'.join(reversed(g)) + ('{,}' + frac if frac else '')


def escreve_tex(res):
    os.makedirs(TEXDIR, exist_ok=True)
    b = res['base']
    ci, cf = res['condicoes']['inicio'], res['condicoes']['fim']
    linhas = [
        r'\begin{tabular}{llr}', r'\toprule',
        r'Parâmetro & Símbolo & Valor \\', r'\midrule',
        r'MTOW & $W_0$ & $%s$ kgf \\' % milhar(b['W0_kgf']),
        r'Peso no início do cruzeiro & $W_{\mathrm{ini}}$ & $%s$ kgf \\'
        % milhar(ci['W_kgf']),
        r'Peso no fim do cruzeiro & $W_{\mathrm{fim}}$ & $%s$ kgf \\'
        % milhar(cf['W_kgf']),
        r'Área de referência & $S_{\mathrm{ref}}$ & $%s$ m$^2$ \\'
        % br(b['Sref'], 2),
        r'Corda de referência (CMA) & $c_{\mathrm{ref}}$ & $%s$ m \\'
        % br(b['cref'], 3),
        r'Altitude de cruzeiro & $h$ & $%s$ ft ($%s$ m) \\'
        % (milhar(b['h_ft']), milhar(b['h_m'])),
        r'Máximo coeficiente de sustentação & $C_{L_{\max}}$ & $%s$ \\'
        % br(b['CLmax'], 3),
        r'Mínimo coeficiente de sustentação & $C_{L_{\min}}$ & $%s$ \\'
        % br(b['CLmin'], 3),
        r'Inclinação da curva de sustentação ($M = %s$) & $C_{L_\alpha}$ & '
        r'$%s$ rad$^{-1}$ \\' % (br(b['MC'], 2), br(b['CLa'], 3)),
        r'Inclinação da curva de sustentação ($M = %s$) & $C_{L_\alpha}$ & '
        r'$%s$ rad$^{-1}$ \\' % (br(b['M_cruzeiro'], 2),
                                  br(b['CLa_cruzeiro'], 3)),
        r'Mach de cruzeiro & $M_{\mathrm{cr}}$ & $%s$ \\'
        % br(b['M_cruzeiro'], 2),
        r'Mach de projeto de cruzeiro (MMO) & $M_C$ & $%s$ \\'
        % br(b['MC'], 2),
        r'Mach de mergulho & $M_D$ & $%s$ \\' % br(b['MD'], 2),
        r'Velocidade de rajada de referência & $U_{\mathrm{ref}}$ & '
        r'$%s$ m/s EAS ($%s$ ft/s) \\'
        % (br(b['U_ref'], 2), br(b['U_ref']/ft2m, 2)),
        r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(TEXDIR, 'tab_dados.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n')

    lin = [r'\begin{tabular}{l rrr rrr}', r'\toprule',
           r' & \multicolumn{3}{c}{Início do cruzeiro} & '
           r'\multicolumn{3}{c}{Fim do cruzeiro} \\',
           r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
           r'Ponto & EAS [m/s] & TAS [m/s] & Mach & EAS [m/s] & TAS [m/s] '
           r'& Mach \\', r'\midrule']
    for k, rot in (('VS1', r'$V_{S1}$'), ('VS_neg', r'$V_{S}^{-}$ ($n=-1$)'),
                   ('VA', r'$V_A$'), ('VB', r'$V_B$'), ('VC', r'$V_C$'),
                   ('VD', r'$V_D$')):
        lin.append(rot + ''.join(
            ' & $%s$ & $%s$ & $%s$' % (br(c['velocidades_EAS'][k], 1),
                                       br(c['velocidades_TAS'][k], 1),
                                       br(c['mach'][k], 3))
            for c in (ci, cf)) + r' \\')
    lin += [r'\midrule']
    for k, rot, nd in (('n_max', r'$n_{\max}$', 2), ('n_min', r'$n_{\min}$', 2),
                       ('WS', r'$W/S$ [Pa]', 0), ('mu_g', r'$\mu_g$', 1),
                       ('K_g', r'$K_g$', 3)):
        lin.append(rot + ''.join(r' & \multicolumn{3}{c}{$%s$}'
                                 % br(c[k], nd) for c in (ci, cf)) + r' \\')
    for k, rot in (('dn_B', r'$n$ de rajada em $V_B$'),
                   ('dn_C', r'$n$ de rajada em $V_C$'),
                   ('dn_D', r'$n$ de rajada em $V_D$')):
        lin.append(rot + ''.join(
            r' & \multicolumn{3}{c}{$%s$ / $%s$}'
            % (br(1 + c[k], 2), br(1 - c[k], 2)) for c in (ci, cf)) + r' \\')
    lin += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(TEXDIR, 'tab_velocidades.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lin) + '\n')


def escreve_csv(res, pasta=RESDIR):
    with open(os.path.join(pasta, 'q1_pontos_vn.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condicao', 'ponto', 'VE_m_s', 'VT_m_s', 'Mach', 'VE_kt'])
        for chave, c in res['condicoes'].items():
            for k, VE in c['velocidades_EAS'].items():
                w.writerow([chave, k, '%.3f' % VE,
                            '%.3f' % c['velocidades_TAS'][k],
                            '%.4f' % c['mach'][k], '%.2f' % (VE/KT)])


def imprime(res):
    b = res['base']
    print('Dados do envelope (NJ-0502, my_airplane vigente):')
    print('  MTOW            %12.1f kgf' % b['W0_kgf'])
    print('  S_ref           %12.3f m2   c_ref %.4f m   c_mg %.4f m'
          % (b['Sref'], b['cref'], b['c_mg']))
    print('  h               %12.0f ft   rho %.4f  a %.2f m/s'
          % (b['h_ft'], b['rho'], b['a']))
    print('  CL_max          %12.4f      (%s)' % (b['CLmax'], b['CLmax_fonte']))
    print('  CL_min          %12.4f' % b['CLmin'])
    print('  CL_alpha        %12.4f /rad (%.5f /deg, AVL M=%.2f, it=%.3f, '
          'washout %.1f)' % (b['CLa'], b['CLa_deg'], b['MC'], b['it_deg'],
                             b.get('washout_deg', 0.0)))
    print('  CL_alpha cruz.  %12.4f /rad (M=%.2f)'
          % (b['CLa_cruzeiro'], b['M_cruzeiro']))
    print('  CL_max AVL      %12.4f      (secao critica, mesma asa)'
          % b['CLmax_avl_q4'])
    print('  M_cr / M_C / M_D %11.3f / %.3f / %.3f'
          % (b['M_cruzeiro'], b['MC'], b['MD']))
    print('  U_ref           %12.3f m/s EAS (%.2f ft/s)'
          % (b['U_ref'], b['U_ref']/ft2m))
    for chave, c in res['condicoes'].items():
        print('\n%s: W = %.1f kgf  W/S = %.1f Pa  n = [%.2f, %.2f]'
              % (ROTULO[chave], c['W_kgf'], c['WS'], c['n_min'], c['n_max']))
        print('  mu_g = %.2f  K_g = %.4f  n_rajada: B %.3f  C %.3f  D %.3f'
              % (c['mu_g'], c['K_g'], 1 + c['dn_B'], 1 + c['dn_C'],
                 1 + c['dn_D']))
        for k, VE in c['velocidades_EAS'].items():
            print('  %-7s EAS %7.2f m/s  TAS %7.2f m/s  M %.4f'
                  % (k, VE, c['velocidades_TAS'][k], c['mach'][k]))
        if not c['VA_existe']:
            print('  ATENCAO: V_A > V_D, a linha de estol nao alcanca n_max')
        elif c['VA_maior_que_VC']:
            print('  obs.: V_A > V_C (permitido; 25.335(c) so diz que V_A '
                  'nao precisa exceder V_C)')


def variante_clmax(res, arg):
    '''Refaz o envelope com outro C_L_max, sem rodar AVL nem designTool.

    arg: numero (ex.: 1.3) ou 'avl' (secao critica do Lab 04 para a asa do
    fwd.avl). C_L_alpha, pesos e Machs vem do JSON oficial; C_L_min segue
    o mesmo CLMIN. Saidas em resultados_q1/clmax_<tag>/.
    '''
    base = dict(res['base'])
    if arg.lower() == 'avl':
        base['CLmax'] = base['CLmax_avl_q4']
        base['CLmax_fonte'] = 'AVL, secao critica (Lab 04, CG dianteiro trimado)'
        tag = 'avl'
    else:
        base['CLmax'] = float(arg)
        base['CLmax_fonte'] = 'fixado manualmente'
        tag = ('%.2f' % base['CLmax']).replace('.', 'p')
    base['CLmin'] = CLMIN
    W = {k: c['W_N'] for k, c in res['condicoes'].items()}
    var = {'base': base,
           'condicoes': {k: envelope(W[k], base) for k in ('inicio', 'fim')}}
    pasta = os.path.join(RESDIR, 'clmax_' + tag)
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, 'q1_envelope.json'), 'w',
              encoding='utf-8') as f:
        json.dump(var, f, indent=2)
    imprime(var)
    for chave in ('inicio', 'fim'):
        figura(var, chave, os.path.join(pasta, 'vn_%s.png' % chave))
    escreve_csv(var, pasta)
    print('\nsaidas em', pasta)


def main():
    os.makedirs(RESDIR, exist_ok=True)
    if '--clmax' in sys.argv:
        # variante: python q1_envelope_voo.py --clmax 1.3 | --clmax avl
        res = json.load(open(JSON_OUT, encoding='utf-8'))
        variante_clmax(res, sys.argv[sys.argv.index('--clmax') + 1])
        return
    if '--tex' in sys.argv:
        res = json.load(open(JSON_OUT, encoding='utf-8'))
    else:
        res = calcula()
        with open(JSON_OUT, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2)
    imprime(res)
    for chave in ('inicio', 'fim'):
        figura(res, chave, os.path.join(RESDIR, 'vn_%s.png' % chave))
    escreve_csv(res)
    escreve_tex(res)
    print('\nsaidas em', RESDIR, 'e', TEXDIR)


if __name__ == '__main__':
    main()
