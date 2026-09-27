'''
PRJ-23 Lab 04 - Secao 3: derivadas de estabilidade (CG traseiro).

Segue o roteiro do enunciado e o molde do item 5: o corpo e a asa reta
(aft.avl oficial); a torcao de -7 deg entra so como subsecao de comparacao,
com arquivos extras aft_washout.avl / fwd_washout.avl.

Duas sessoes AVL por geometria, sem restricao de trimagem (d4 d4 0):
  1. alpha = 0, i_t = 0          -> CL0, CM0 (menu ft)
  2. ponto de projeto (a c CL, i_t do caso) -> ft + st + sb

Conversoes do enunciado (MVFO):
  - derivadas de controle e de i_t do AVL vem em 1/grau -> vezes 180/pi
  - demais (alpha, p, q, r, beta) ja estao em 1/rad
  - inverter CYbeta, CYp, CYr, Clda, Cldr, Cnda, Cndr
  - momentos latero-direcionais (Cl*, Cn*) saem do sb; o resto do st
  - z_p do designTool e invertido (MVFO: z positivo para baixo)
  - CDq, CDde e CDit nao vem no st: monta-se CD = -CX cos a - CZ sin a
    a partir do sb

Uso (nesta pasta):
    python s3_estabilidade.py         # AVL + tabelas + figura
    python s3_estabilidade.py --tex   # so reescreve saidas
'''

import json
import os
import shutil
import subprocess
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

import avl_tools as at                                 # noqa: E402
import ponto_projeto as pj                             # noqa: E402
import opt_common as oc                                # noqa: E402
from designTool.constants import gravity               # noqa: E402
from designTool.moment_of_inertia import moment_of_inertia  # noqa: E402

OUT = os.path.join(LAB, 'resultados_s3')
TEX = os.path.join(LAB, 'tex_s3')
Q5 = os.path.join(LAB, 'resultados_q5', 'q5_polares.json')
WO = os.path.join(LAB, 'resultados_washout', 'q4_washout.json')

TWIST_TIP = -7.0
DEG2RAD = 180.0 / np.pi

# sinais do enunciado (depois da conversao de unidade, se houver)
INVERTE = {'CYb', 'CYp', 'CYr', 'Cld3', 'Cld5', 'Cnd3', 'Cnd5'}


# ---------------------------------------------------------------- AVL extras

def escreve_avl_torcido(origem, destino, twist_tip, b2):
    '''Copia um .avl com Ainc linear 0 na raiz a twist_tip na ponta (so Wing).'''
    linhas = open(os.path.join(at.AVLDIR, origem),
                  encoding='utf-8').read().splitlines()
    saida, surf, k = [], None, 0
    while k < len(linhas):
        ln = linhas[k]
        if k == 0:
            saida.append(ln + ' | washout %+.1f deg (extra, nao oficial)'
                         % twist_tip)
            k += 1
            continue
        saida.append(ln)
        if ln.strip().upper() == 'SURFACE':
            surf = linhas[k + 1].strip()
        if ln.strip().upper() == 'SECTION' and surf == 'Wing':
            j = k + 1
            while not linhas[j].strip():
                saida.append(linhas[j])
                j += 1
            campos = linhas[j].split()
            campos[4] = '%.4f' % (twist_tip * float(campos[1]) / b2)
            saida.append(' '.join(campos))
            k = j + 1
            continue
        k += 1
    caminho = os.path.join(at.AVLDIR, destino)
    with open(caminho, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(saida) + '\n')
    return destino


# ---------------------------------------------------------------- sessao AVL

def _script(avl, mach, it, modo, cl, tag):
    linhas = ['load %s' % avl, 'oper', 'm', 'mn %.6f' % mach, '']
    if modo == 'alpha0':
        linhas.append('a a 0.0')
    else:
        linhas.append('a c %.6f' % cl)
    linhas += ['d4 d4 0.0', 'de', '1 %.6f' % it, '',
               'x',
               'ft', '%s_ft.txt' % tag,
               'st', '%s_st.txt' % tag,
               'sb', '%s_sb.txt' % tag,
               '', 'quit', '']
    return '\n'.join(linhas)


def roda_sessao(avl, mach, it, modo, cl, tag):
    wd = os.path.join(LAB, '_avl_tmp', '%d_%s' % (os.getpid(), tag))
    os.makedirs(wd, exist_ok=True)
    for name in os.listdir(at.AVLDIR):
        if name.endswith(('.avl', '.dat', '.exe')):
            dst = os.path.join(wd, name)
            src = os.path.join(at.AVLDIR, name)
            if (not os.path.exists(dst)
                    or os.path.getmtime(src) > os.path.getmtime(dst)):
                shutil.copy2(src, dst)
    proc = subprocess.run([os.path.join(wd, 'avl337.exe')],
                          input=_script(avl, mach, it, modo, cl, tag),
                          capture_output=True, text=True, cwd=wd, timeout=180)
    caminhos = {k: os.path.join(wd, '%s_%s.txt' % (tag, k))
                for k in ('ft', 'st', 'sb')}
    for k, p in caminhos.items():
        if not os.path.isfile(p):
            raise RuntimeError('AVL nao gravou %s (%s)\n%s'
                               % (k, tag, proc.stdout[-2000:]))
    ft = at.read_ft(caminhos['ft'])
    st = at.read_derivs(caminhos['st'])
    sb = at.read_derivs(caminhos['sb'])
    raw = {k: open(caminhos[k], encoding='utf-8', errors='replace').read()
           for k in caminhos}
    shutil.rmtree(wd, ignore_errors=True)
    return {'ft': ft, 'st': st, 'sb': sb, 'raw': raw,
            'it_deg': float(it), 'modo': modo}


def cd_de_eixo(sb, suf, alpha_deg):
    '''CD = -CX cos a - CZ sin a (eixos de geometria do AVL, X fwd Z down).'''
    a = np.radians(alpha_deg)
    cx = sb.get('CX' + suf, float('nan'))
    cz = sb.get('CZ' + suf, float('nan'))
    return float(-cx * np.cos(a) - cz * np.sin(a))


def deriva(st, sb, ft, chave, fonte='st', por_grau=False, inverter=False):
    tab = {'st': st, 'sb': sb, 'ft': ft}[fonte]
    val = float(tab[chave])
    if por_grau:
        val *= DEG2RAD
    if inverter:
        val = -val
    return val


def monta_derivadas(sessao):
    '''Aplica as conversoes do enunciado sobre uma sessao de ponto de projeto.'''
    st, sb, ft = sessao['st'], sessao['sb'], sessao['ft']
    a = float(ft['Alpha'])
    d = {
        'alpha_deg': a,
        'it_deg': float(sessao['it_deg']),
        'delta_e_deg': float(ft.get('elevator', 0.0)),
        'CL': float(ft['CLtot']),
        'CD': float(ft['CDtot']),
        'CM': float(ft['Cmtot']),
        'Xnp': float(st.get('Xnp', float('nan'))),
        'CLa': deriva(st, sb, ft, 'CLa', 'st'),
        'CLq': deriva(st, sb, ft, 'CLq', 'st'),
        'CLg1': deriva(st, sb, ft, 'CLg1', 'st', por_grau=True),
        'CLd4': deriva(st, sb, ft, 'CLd4', 'st', por_grau=True),
        'Cma': deriva(st, sb, ft, 'Cma', 'st'),
        'Cmq': deriva(st, sb, ft, 'Cmq', 'st'),
        'Cmg1': deriva(st, sb, ft, 'Cmg1', 'st', por_grau=True),
        'Cmd4': deriva(st, sb, ft, 'Cmd4', 'st', por_grau=True),
        'CYb': deriva(st, sb, ft, 'CYb', 'st', inverter=True),
        'CYp': deriva(st, sb, ft, 'CYp', 'st', inverter=True),
        'CYr': deriva(st, sb, ft, 'CYr', 'st', inverter=True),
        'CYd5': deriva(st, sb, ft, 'CYd5', 'st', por_grau=True),
        # Clb/Cnb: o sb reporta Clv/Cnv (eixo de geometria), nao Clb/Cnb.
        # O enunciado manda o sb so para p, r e comandos laterais.
        'Clb': deriva(st, sb, ft, 'Clb', 'st'),
        'Clp': deriva(st, sb, ft, 'Clp', 'sb'),
        'Clr': deriva(st, sb, ft, 'Clr', 'sb'),
        'Cld3': deriva(st, sb, ft, 'Cld3', 'sb', por_grau=True, inverter=True),
        'Cld5': deriva(st, sb, ft, 'Cld5', 'sb', por_grau=True, inverter=True),
        'Cnb': deriva(st, sb, ft, 'Cnb', 'st'),
        'Cnp': deriva(st, sb, ft, 'Cnp', 'sb'),
        'Cnr': deriva(st, sb, ft, 'Cnr', 'sb'),
        'Cnd3': deriva(st, sb, ft, 'Cnd3', 'sb', por_grau=True, inverter=True),
        'Cnd5': deriva(st, sb, ft, 'Cnd5', 'sb', por_grau=True, inverter=True),
        'CDq': cd_de_eixo(sb, 'q', a),
        'CDg1': cd_de_eixo(sb, 'g1', a) * DEG2RAD,
        'CDd4': cd_de_eixo(sb, 'd4', a) * DEG2RAD,
    }
    return d


# ---------------------------------------------------------------- dados DT

def dados_gerais(ap, pp):
    I, T = ap['inputs'], ap['thrust_matching']
    W_mid = pp['W_N']
    W_empty = float(T['W_empty'])
    W_fuel = float(T['W_fuel'])
    W_pay = float(I['W_payload'])
    W_crew = float(I['W_crew'])
    fuel_frac = (W_mid - W_empty - W_pay - W_crew) / W_fuel
    moment_of_inertia(ap, fuel_frac=fuel_frac, payload_frac=1.0)
    moi = ap['moment_of_inertia']
    Tmax = float(I['engine']['Tmax'] * I['n_engines'])
    return {
        'Sref': pp['Sref'], 'cref': pp['cref'], 'bref': pp['bref'],
        'm_kg': W_mid / gravity,
        'W_N': W_mid,
        'fuel_frac': float(fuel_frac),
        'Ixx': float(moi['Ixx']), 'Iyy': float(moi['Iyy']),
        'Izz': float(moi['Izz']), 'Ixz': float(moi['Ixz']),
        'ip_deg': 0.0,
        'xp_m': float(I['x_n']),
        'zp_m': float(-I['z_n']),
        'z_n_DT': float(I['z_n']),
        'Tmax_N': Tmax,
        'T0_N': float(T['T0']),
        'V': pp['V'], 'h_m': pp['h_m'], 'M': pp['M'], 'CL': pp['CL'],
        'xcg_aft': pp['xcg_aft'],
    }


def ajuste_de(json_path, chave_ajuste):
    with open(json_path, encoding='utf-8') as f:
        j = json.load(f)
    if chave_ajuste == 'q5':
        a = j['ajuste_quadratico']
        pol = j['polares']['aft_livre']
        return {'CD0': a['CD0'], 'CDa': a['CD_alpha_por_rad'],
                'CDaa': a['CD_alpha2_por_rad2'], 'r2': a['r2'],
                'alpha': pol['alpha'], 'CD': pol['CD'], 'CL': pol['CL']}
    a = j['polares']['ajuste']
    pol = j['polares']['casos']['aft_livre']
    return {'CD0': a['CD0'], 'CDa': a['CD_alpha_por_rad'],
            'CDaa': a['CD_alpha2_por_rad2'], 'r2': a['r2'],
            'alpha': pol['alpha'], 'CD': pol['CD'], 'CL': pol.get('CL')}


# ---------------------------------------------------------------- nucleo

def calcula():
    ap = oc.run_designTool(oc.get_baseline())
    pp = pj.ponto_projeto(ap)
    its = pj.incidencias_eh()
    it_reta = its['aft']['it_deg']
    with open(WO, encoding='utf-8') as f:
        wo = json.load(f)
    it_torc = wo['casos']['-7.0']['it_aft']
    b2 = pp['bref'] / 2.0

    print('gravando aft_washout.avl e fwd_washout.avl (Ainc 0 -> %+.1f deg)'
          % TWIST_TIP)
    escreve_avl_torcido('aft.avl', 'aft_washout.avl', TWIST_TIP, b2)
    escreve_avl_torcido('fwd.avl', 'fwd_washout.avl', TWIST_TIP, b2)

    gerais = dados_gerais(ap, pp)
    print('m = %.0f kg  fuel_frac = %.3f  Ixx = %.3e'
          % (gerais['m_kg'], gerais['fuel_frac'], gerais['Ixx']))

    configs = {
        'reta': {'avl': 'aft.avl', 'it': it_reta, 'rotulo': 'asa reta'},
        'torcida': {'avl': 'aft_washout.avl', 'it': it_torc,
                    'rotulo': r'asa torcida ($\varepsilon_t=-7^\circ$)'},
    }
    resultados = {}
    for nome, cfg in configs.items():
        print('AVL %s  (%s, it = %+.3f deg)' % (nome, cfg['avl'], cfg['it']))
        ref = roda_sessao(cfg['avl'], pp['M'], 0.0, 'alpha0', pp['CL'],
                          's3_%s_ref' % nome)
        proj = roda_sessao(cfg['avl'], pp['M'], cfg['it'], 'projeto', pp['CL'],
                           's3_%s_proj' % nome)
        resultados[nome] = {
            'rotulo': cfg['rotulo'], 'avl': cfg['avl'], 'it_deg': cfg['it'],
            'CL0': float(ref['ft']['CLtot']),
            'CM0': float(ref['ft']['Cmtot']),
            'CD_ref': float(ref['ft']['CDtot']),
            'derivadas': monta_derivadas(proj),
            'Xnp_ref': float(ref['st'].get('Xnp', float('nan'))),
        }
        d = resultados[nome]['derivadas']
        print('  CL0=%.4f  CM0=%.4f  |  proj CL=%.4f  alpha=%.3f  '
              'CLa=%.3f  Cma=%.3f' % (
                  resultados[nome]['CL0'], resultados[nome]['CM0'],
                  d['CL'], d['alpha_deg'], d['CLa'], d['Cma']))

    aj_reta = ajuste_de(Q5, 'q5')
    aj_torc = ajuste_de(WO, 'wo')
    return {'ponto_projeto': pp, 'gerais': gerais,
            'ajuste_reta': {k: aj_reta[k] for k in
                            ('CD0', 'CDa', 'CDaa', 'r2')},
            'ajuste_torcida': {k: aj_torc[k] for k in
                               ('CD0', 'CDa', 'CDaa', 'r2')},
            'polar_reta': {'alpha': aj_reta['alpha'], 'CD': aj_reta['CD']},
            'polar_torcida': {'alpha': aj_torc['alpha'], 'CD': aj_torc['CD']},
            'configs': resultados,
            'twist_tip': TWIST_TIP}


# ---------------------------------------------------------------- saidas

def _fmt(x, fmt):
    return fmt % x


def escreve_saidas(dados):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)
    g, r, t = dados['gerais'], dados['configs']['reta'], dados['configs']['torcida']
    ar, atorc = dados['ajuste_reta'], dados['ajuste_torcida']
    dr, dt = r['derivadas'], t['derivadas']

    dump = json.loads(json.dumps(dados))
    with open(os.path.join(OUT, 's3_estabilidade.json'), 'w',
              encoding='utf-8') as f:
        json.dump(dump, f, indent=2)

    # figura do ajuste (item 5.b, asa reta) + a da torcida por cima
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for nome, cor, mk in (('polar_reta', 'tab:blue', 'o'),
                          ('polar_torcida', 'tab:red', 's')):
        pol = dados[nome]
        a = np.array(pol['alpha'])
        cd = np.array(pol['CD'])
        aj = dados['ajuste_reta' if nome.endswith('reta') else 'ajuste_torcida']
        rot = 'asa reta (item 5.b)' if nome.endswith('reta') else r'$\varepsilon_t=-7^\circ$'
        ax.plot(a, cd, mk, ms=4.5, color=cor, mfc='none', mew=1.1, label=rot)
        aa = np.linspace(a.min(), a.max(), 250)
        ax.plot(aa, np.polyval([aj['CDaa'], aj['CDa'], aj['CD0']],
                               np.radians(aa)), '-', color=cor, lw=1.6)
    ax.set_xlabel(r'$\alpha$ [graus]')
    ax.set_ylabel(r'$C_D$')
    ax.set_title(r'Ajuste $C_D=C_{D0}+C_{D\alpha}\alpha+C_{D\alpha^2}\alpha^2$'
                 r' (CG traseiro, sem trimagem)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ajuste_polar_quadratica.png'), dpi=200)
    plt.close(fig)

    def tab_gerais(path):
        linhas = [
            (r'$S_{\mathrm{ref}}$', r'Area de referencia [m$^2$]',
             '%.4f' % g['Sref']),
            (r'$c_{\mathrm{ref}}$', r'Corda de referencia [m]',
             '%.4f' % g['cref']),
            (r'$b_{\mathrm{ref}}$', r'Envergadura de referencia [m]',
             '%.4f' % g['bref']),
            (r'$m$', r'Massa da aeronave [kg] (meio do cruzeiro)',
             '%.0f' % g['m_kg']),
            (r'$I_{xx}$', r'Momento de inercia [kg$\cdot$m$^2$]',
             '%.3e' % g['Ixx']),
            (r'$I_{yy}$', r'Momento de inercia [kg$\cdot$m$^2$]',
             '%.3e' % g['Iyy']),
            (r'$I_{zz}$', r'Momento de inercia [kg$\cdot$m$^2$]',
             '%.3e' % g['Izz']),
            (r'$I_{xz}$', r'Momento de inercia [kg$\cdot$m$^2$]',
             '%.3e' % g['Ixz']),
            (r'$i_p$', r'Incidencia do motor [$^\circ$]',
             '%.1f' % g['ip_deg']),
            (r'$x_p$', r'Posicao longitudinal do motor [m]',
             '%.3f' % g['xp_m']),
            (r'$z_p$', r'Posicao vertical do motor [m] (sinal invertido)',
             '%.3f' % g['zp_m']),
            (r'$T_{\max}$', r'Tracao maxima [N]',
             '%.0f' % g['Tmax_N']),
            (r'$V$', r'Velocidade de voo [m/s] (ponto de projeto)',
             '%.2f' % g['V']),
            (r'$h$', r'Altitude de voo [m] (ponto de projeto)',
             '%.0f' % g['h_m']),
        ]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\\begin{tabular}{llc}\n\\toprule\n')
            f.write('Parametro & Descricao & Valor \\\\\n\\midrule\n')
            for p, desc, val in linhas:
                f.write('%s & %s & $%s$ \\\\\n' % (p, desc, val))
            f.write('\\bottomrule\n\\end{tabular}\n')

    tab_gerais(os.path.join(TEX, 'tab_dados_gerais.tex'))

    with open(os.path.join(TEX, 'tab_cl0_cm0.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{llc}\n\\toprule\n')
        f.write('Parametro & Descricao & Valor \\\\\n\\midrule\n')
        f.write(r'$C_{L0}$ & $C_L$ para $\alpha=0$ & $%.5f$ \\' % r['CL0']
                + '\n')
        f.write(r'$C_{M0}$ & $C_M$ para $\alpha=0$ & $%.5f$ \\' % r['CM0']
                + '\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(os.path.join(TEX, 'tab_ajuste.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{llc}\n\\toprule\n')
        f.write('Parametro & Descricao & Valor \\\\\n\\midrule\n')
        f.write(r'$C_{D0}$ & Termo constante da polar de arrasto & $%.5f$ \\'
                % ar['CD0'] + '\n')
        f.write(r'$C_{D\alpha}$ & Termo linear da polar [1/rad] & $%.5f$ \\'
                % ar['CDa'] + '\n')
        f.write(r'$C_{D\alpha^2}$ & Termo quadratico da polar [1/rad$^2$] & '
                r'$%.5f$ \\' % ar['CDaa'] + '\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    linhas_der = [
        (r'$S_{\mathrm{ref}}$', r'Area de referencia [m$^2$]',
         '%.4f' % g['Sref']),
        (r'$c_{\mathrm{ref}}$', r'Corda de referencia [m]',
         '%.4f' % g['cref']),
        (r'$b_{\mathrm{ref}}$', r'Envergadura de referencia [m]',
         '%.4f' % g['bref']),
        (r'$m$', r'Massa da aeronave [kg]', '%.0f' % g['m_kg']),
        (r'$I_{xx}$', r'Momento de inercia [kg$\cdot$m$^2$]',
         '%.3e' % g['Ixx']),
        (r'$I_{yy}$', r'Momento de inercia [kg$\cdot$m$^2$]',
         '%.3e' % g['Iyy']),
        (r'$I_{zz}$', r'Momento de inercia [kg$\cdot$m$^2$]',
         '%.3e' % g['Izz']),
        (r'$I_{xz}$', r'Momento de inercia [kg$\cdot$m$^2$]',
         '%.3e' % g['Ixz']),
        (r'$i_p$', r'Incidencia do motor [$^\circ$]', '%.1f' % g['ip_deg']),
        (r'$x_p$', r'Posicao longitudinal do motor [m]', '%.3f' % g['xp_m']),
        (r'$z_p$', r'Posicao vertical do motor [m] (sinal invertido)',
         '%.3f' % g['zp_m']),
        (r'$T_{\max}$', r'Tracao maxima [N]', '%.0f' % g['Tmax_N']),
        (r'$V$', r'Velocidade de voo [m/s]', '%.2f' % g['V']),
        (r'$h$', r'Altitude de voo [m]', '%.0f' % g['h_m']),
        (r'$C_{L0}$', r'$C_L$ para $\alpha=0$', '%.5f' % r['CL0']),
        (r'$C_{L\alpha}$', r'$\partial C_L/\partial\alpha$ [1/rad]',
         '%.5f' % dr['CLa']),
        (r'$C_{Lq}$', r'$\partial C_L/\partial q$ [1/rad]', '%.5f' % dr['CLq']),
        (r'$C_{Li_t}$', r'$\partial C_L/\partial i_t$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['CLg1']),
        (r'$C_{L\delta_e}$',
         r'$\partial C_L/\partial\delta_e$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['CLd4']),
        (r'$C_{D0}$', r'$C_D$ para $\alpha=0$ (ajuste 5.b)', '%.5f' % ar['CD0']),
        (r'$C_{D\alpha}$', r'Termo linear da polar [1/rad]', '%.5f' % ar['CDa']),
        (r'$C_{D\alpha^2}$', r'Termo quadratico da polar [1/rad$^2$]',
         '%.5f' % ar['CDaa']),
        (r'$C_{Dq}$', r'$\partial C_D/\partial q$ [1/rad]', '%.5f' % dr['CDq']),
        (r'$C_{Di_t}$',
         r'$\partial C_D/\partial i_t$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['CDg1']),
        (r'$C_{D\delta_e}$',
         r'$\partial C_D/\partial\delta_e$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['CDd4']),
        (r'$C_{M0}$', r'$C_M$ para $\alpha=0$', '%.5f' % r['CM0']),
        (r'$C_{M\alpha}$', r'$\partial C_M/\partial\alpha$ [1/rad]',
         '%.5f' % dr['Cma']),
        (r'$C_{Mq}$', r'$\partial C_M/\partial q$ [1/rad]', '%.5f' % dr['Cmq']),
        (r'$C_{Mi_t}$',
         r'$\partial C_M/\partial i_t$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['Cmg1']),
        (r'$C_{M\delta_e}$',
         r'$\partial C_M/\partial\delta_e$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['Cmd4']),
        (r'$C_{Y\beta}$',
         r'$\partial C_Y/\partial\beta$ [1/rad] (sinal invertido)',
         '%.5f' % dr['CYb']),
        (r'$C_{Yp}$',
         r'$\partial C_Y/\partial p$ [1/rad] (sinal invertido)',
         '%.5f' % dr['CYp']),
        (r'$C_{Yr}$',
         r'$\partial C_Y/\partial r$ [1/rad] (sinal invertido)',
         '%.5f' % dr['CYr']),
        (r'$C_{Y\delta_r}$',
         r'$\partial C_Y/\partial\delta_r$ [1/rad] (mult.\ $180/\pi$)',
         '%.5f' % dr['CYd5']),
        (r'$C_{\ell\beta}$', r'$\partial C_\ell/\partial\beta$ [1/rad]',
         '%.5f' % dr['Clb']),
        (r'$C_{\ell p}$', r'$\partial C_\ell/\partial p$ [1/rad]',
         '%.5f' % dr['Clp']),
        (r'$C_{\ell r}$', r'$\partial C_\ell/\partial r$ [1/rad]',
         '%.5f' % dr['Clr']),
        (r'$C_{\ell\delta_a}$',
         r'$\partial C_\ell/\partial\delta_a$ [1/rad] (inv., $180/\pi$)',
         '%.5f' % dr['Cld3']),
        (r'$C_{\ell\delta_r}$',
         r'$\partial C_\ell/\partial\delta_r$ [1/rad] (inv., $180/\pi$)',
         '%.5f' % dr['Cld5']),
        (r'$C_{n\beta}$', r'$\partial C_n/\partial\beta$ [1/rad]',
         '%.5f' % dr['Cnb']),
        (r'$C_{np}$', r'$\partial C_n/\partial p$ [1/rad]',
         '%.5f' % dr['Cnp']),
        (r'$C_{nr}$', r'$\partial C_n/\partial r$ [1/rad]',
         '%.5f' % dr['Cnr']),
        (r'$C_{n\delta_a}$',
         r'$\partial C_n/\partial\delta_a$ [1/rad] (inv., $180/\pi$)',
         '%.5f' % dr['Cnd3']),
        (r'$C_{n\delta_r}$',
         r'$\partial C_n/\partial\delta_r$ [1/rad] (inv., $180/\pi$)',
         '%.5f' % dr['Cnd5']),
    ]
    with open(os.path.join(TEX, 'tab_derivadas.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{llc}\n\\toprule\n')
        f.write('Parametro & Descricao & Valor \\\\\n\\midrule\n')
        for p, desc, val in linhas_der:
            f.write('%s & %s & $%s$ \\\\\n' % (p, desc, val))
        f.write('\\bottomrule\n\\end{tabular}\n')

    comps = [
        (r'$i_t$ de equilibrio [$^\circ$]', r['it_deg'], t['it_deg']),
        (r'$\alpha$ no ponto [$^\circ$]', dr['alpha_deg'], dt['alpha_deg']),
        (r'$C_D$ no ponto de projeto', dr['CD'], dt['CD']),
        (r'$C_{L0}$ ($\alpha=0$, $i_t=0$)', r['CL0'], t['CL0']),
        (r'$C_{M0}$ ($\alpha=0$, $i_t=0$)', r['CM0'], t['CM0']),
        (r'$C_D$ em $\alpha=0$, $i_t=0$', r['CD_ref'], t['CD_ref']),
        (r'$C_{L\alpha}$', dr['CLa'], dt['CLa']),
        (r'$C_{Lq}$', dr['CLq'], dt['CLq']),
        (r'$C_{Li_t}$', dr['CLg1'], dt['CLg1']),
        (r'$C_{L\delta_e}$', dr['CLd4'], dt['CLd4']),
        (r'$C_{D0}$ (ajuste 5.b)', ar['CD0'], atorc['CD0']),
        (r'$C_{D\alpha}$ (ajuste 5.b)', ar['CDa'], atorc['CDa']),
        (r'$C_{D\alpha^2}$ (ajuste 5.b)', ar['CDaa'], atorc['CDaa']),
        (r'$C_{Dq}$', dr['CDq'], dt['CDq']),
        (r'$C_{Di_t}$', dr['CDg1'], dt['CDg1']),
        (r'$C_{D\delta_e}$', dr['CDd4'], dt['CDd4']),
        (r'$C_{M\alpha}$', dr['Cma'], dt['Cma']),
        (r'$C_{Mq}$', dr['Cmq'], dt['Cmq']),
        (r'$C_{Mi_t}$', dr['Cmg1'], dt['Cmg1']),
        (r'$C_{M\delta_e}$', dr['Cmd4'], dt['Cmd4']),
        (r'$C_{Y\beta}$', dr['CYb'], dt['CYb']),
        (r'$C_{Yp}$', dr['CYp'], dt['CYp']),
        (r'$C_{Yr}$', dr['CYr'], dt['CYr']),
        (r'$C_{Y\delta_r}$', dr['CYd5'], dt['CYd5']),
        (r'$C_{\ell\beta}$', dr['Clb'], dt['Clb']),
        (r'$C_{\ell p}$', dr['Clp'], dt['Clp']),
        (r'$C_{\ell r}$', dr['Clr'], dt['Clr']),
        (r'$C_{\ell\delta_a}$', dr['Cld3'], dt['Cld3']),
        (r'$C_{\ell\delta_r}$', dr['Cld5'], dt['Cld5']),
        (r'$C_{n\beta}$', dr['Cnb'], dt['Cnb']),
        (r'$C_{np}$', dr['Cnp'], dt['Cnp']),
        (r'$C_{nr}$', dr['Cnr'], dt['Cnr']),
        (r'$C_{n\delta_a}$', dr['Cnd3'], dt['Cnd3']),
        (r'$C_{n\delta_r}$', dr['Cnd5'], dt['Cnd5']),
        (r'$X_{np}$ [m]', dr['Xnp'], dt['Xnp']),
        (r'SM AVL [\%]',
         100 * (dr['Xnp'] - g['xcg_aft']) / g['cref'],
         100 * (dt['Xnp'] - g['xcg_aft']) / g['cref']),
    ]
    with open(os.path.join(TEX, 'tab_washout_derivadas.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lrrrl}\n\\toprule\n')
        f.write('Parametro & Asa reta & $\\varepsilon_t=-7^\\circ$ & '
                '$\\Delta$ & $\\Delta$ [\\%] \\\\\n\\midrule\n')
        for nome, a, b in comps:
            delta = b - a
            if a * b < 0:
                difs = 'troca sinal'
            elif abs(a) < 1e-12:
                difs = '---'
            elif abs(a) < 0.05 and abs(b / a) > 10:
                difs = '---'
            else:
                difs = '$%+.2f$' % (100 * (b / a - 1))
            f.write('%s & $%.5f$ & $%.5f$ & $%+.5f$ & %s \\\\\n'
                    % (nome, a, b, delta, difs))
        f.write('\\bottomrule\n\\end{tabular}\n')

    def pct(a, b):
        if a * b < 0:
            return 'troca sinal'
        if abs(a) < 1e-12:
            return '---'
        return '%+.1f' % (100 * (b / a - 1))

    mac = {
        'esMach': '%.2f' % g['M'], 'esCL': '%.4f' % g['CL'],
        'esItReta': '%+.3f' % r['it_deg'],
        'esItTorc': '%+.3f' % t['it_deg'],
        'esCL0': '%.5f' % r['CL0'], 'esCM0': '%.5f' % r['CM0'],
        'esCL0w': '%.5f' % t['CL0'], 'esCM0w': '%.5f' % t['CM0'],
        'esCDref': '%.5f' % r['CD_ref'], 'esCDrefW': '%.5f' % t['CD_ref'],
        'esCDproj': '%.5f' % dr['CD'], 'esCDprojW': '%.5f' % dt['CD'],
        'esIw': '%.1f' % dados['ponto_projeto']['iw_deg'],
        'esFuel': '%.2f' % (100 * g['fuel_frac']),
        'esM': '%.0f' % g['m_kg'],
        'esXnp': '%.3f' % dr['Xnp'],
        'esXnpW': '%.3f' % dt['Xnp'],
        'esXcg': '%.3f' % g['xcg_aft'],
        'esSM': '%.1f' % (100 * (dr['Xnp'] - g['xcg_aft']) / g['cref']),
        'esSMW': '%.1f' % (100 * (dt['Xnp'] - g['xcg_aft']) / g['cref']),
        'esCLa': '%.3f' % dr['CLa'], 'esCLaW': '%.3f' % dt['CLa'],
        'esCLq': '%.3f' % dr['CLq'], 'esCLqW': '%.3f' % dt['CLq'],
        'esCma': '%.3f' % dr['Cma'], 'esCmaW': '%.3f' % dt['Cma'],
        'esCmq': '%.2f' % dr['Cmq'], 'esCmqW': '%.2f' % dt['Cmq'],
        'esClb': '%.3f' % dr['Clb'], 'esClbW': '%.3f' % dt['Clb'],
        'esClp': '%.3f' % dr['Clp'], 'esClpW': '%.3f' % dt['Clp'],
        'esClr': '%.3f' % dr['Clr'], 'esClrW': '%.3f' % dt['Clr'],
        'esCnb': '%.3f' % dr['Cnb'], 'esCnbW': '%.3f' % dt['Cnb'],
        'esCnp': '%.3f' % dr['Cnp'], 'esCnpW': '%.3f' % dt['Cnp'],
        'esCnr': '%.3f' % dr['Cnr'], 'esCnrW': '%.3f' % dt['Cnr'],
        'esCYb': '%.3f' % dr['CYb'], 'esCYbW': '%.3f' % dt['CYb'],
        'esCYp': '%.3f' % dr['CYp'], 'esCYpW': '%.3f' % dt['CYp'],
        'esCYr': '%.3f' % dr['CYr'], 'esCYrW': '%.3f' % dt['CYr'],
        'esCLd4': '%.4f' % dr['CLd4'], 'esCLd4W': '%.4f' % dt['CLd4'],
        'esCmd4': '%.4f' % dr['Cmd4'], 'esCmd4W': '%.4f' % dt['Cmd4'],
        'esCLg1': '%.4f' % dr['CLg1'], 'esCLg1W': '%.4f' % dt['CLg1'],
        'esCmg1': '%.4f' % dr['Cmg1'], 'esCmg1W': '%.4f' % dt['Cmg1'],
        'esCld3': '%.4f' % dr['Cld3'], 'esCld3W': '%.4f' % dt['Cld3'],
        'esCld5': '%.4f' % dr['Cld5'], 'esCld5W': '%.4f' % dt['Cld5'],
        'esCnd3': '%.4f' % dr['Cnd3'], 'esCnd3W': '%.4f' % dt['Cnd3'],
        'esCnd5': '%.4f' % dr['Cnd5'], 'esCnd5W': '%.4f' % dt['Cnd5'],
        'esCYd5': '%.4f' % dr['CYd5'], 'esCYd5W': '%.4f' % dt['CYd5'],
        'esCDq': '%.3f' % dr['CDq'], 'esCDqW': '%.3f' % dt['CDq'],
        'esCDg1': '%.4f' % dr['CDg1'], 'esCDg1W': '%.4f' % dt['CDg1'],
        'esCDd4': '%.4f' % dr['CDd4'], 'esCDd4W': '%.4f' % dt['CDd4'],
        'esAlpha': '%.3f' % dr['alpha_deg'],
        'esAlphaW': '%.3f' % dt['alpha_deg'],
        'esCDaFit': '%.5f' % ar['CDa'], 'esCDaFitW': '%.5f' % atorc['CDa'],
        'esCDaaFit': '%.5f' % ar['CDaa'], 'esCDaaFitW': '%.5f' % atorc['CDaa'],
        'esCD0Fit': '%.5f' % ar['CD0'], 'esCD0FitW': '%.5f' % atorc['CD0'],
        'esRdois': '%.5f' % ar['r2'], 'esRdoisW': '%.5f' % atorc['r2'],
        'esZp': '%.3f' % g['zp_m'],
        'esZnDT': '%.3f' % g['z_n_DT'],
        'esXp': '%.3f' % g['xp_m'],
        'esTwist': '%+.0f' % dados['twist_tip'],
        'esPctCLa': pct(dr['CLa'], dt['CLa']),
        'esPctCLq': pct(dr['CLq'], dt['CLq']),
        'esPctCma': pct(dr['Cma'], dt['Cma']),
        'esPctClb': pct(dr['Clb'], dt['Clb']),
        'esPctClr': pct(dr['Clr'], dt['Clr']),
        'esPctCYp': pct(dr['CYp'], dt['CYp']),
        'esPctCYr': pct(dr['CYr'], dt['CYr']),
        'esPctCnp': pct(dr['Cnp'], dt['Cnp']),
        'esPctCnb': pct(dr['Cnb'], dt['Cnb']),
        'esPctCDq': pct(dr['CDq'], dt['CDq']),
        'esPctCDg1': pct(dr['CDg1'], dt['CDg1']),
        'esPctCL0': pct(r['CL0'], t['CL0']),
        'esPctCDproj': pct(dr['CD'], dt['CD']),
    }
    with open(os.path.join(TEX, 'macros.tex'), 'w', encoding='utf-8') as f:
        f.write('% gerado por s3_estabilidade.py\n')
        for k in sorted(mac):
            f.write('\\newcommand{\\%s}{%s}\n' % (k, mac[k]))

    print('gravado em', OUT)
    print('tabelas em', TEX)


def main(tex_only=False):
    for d in (OUT, TEX):
        os.makedirs(d, exist_ok=True)
    caminho = os.path.join(OUT, 's3_estabilidade.json')
    if tex_only:
        with open(caminho, encoding='utf-8') as f:
            dados = json.load(f)
        print('reaproveitando', caminho)
    else:
        dados = calcula()
    escreve_saidas(dados)


if __name__ == '__main__':
    main(tex_only=('--tex' in sys.argv))
