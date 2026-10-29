'''
PRJ-23 Atividade 06 - Item 2: AVL em V_A e n_max, inicio e fim do cruzeiro.

Usa o envelope do item 1 (CLmax do designTool) e o fwd.avl oficial
(washout -7 deg). Aeronave trimada (CM = 0 no profundor) e com taxa de
arfagem q = g (n_z - 1) / V_TAS, prescrita no AVL como qc/2V.

Uso (em designTool/lab06_cargas):
    python q2_va_manobra.py         # AVL + figura + JSON + arquivo do relatorio
    python q2_va_manobra.py --tex   # so refaz tex_q2/item2_resultados.tex
'''

import csv
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNTOOL_DIR = os.path.abspath(os.path.join(HERE, '..'))
LAB04 = os.path.join(DESIGNTOOL_DIR, 'lab04_aero')
RESDIR = os.path.join(HERE, 'resultados_q2')
TEXDIR = os.path.join(HERE, 'tex_q2')
JSON_Q1 = os.path.join(HERE, 'resultados_q1', 'q1_envelope.json')
JSON_Q2 = os.path.join(RESDIR, 'q2_va_manobra.json')
FIG_Q2 = os.path.join(RESDIR, 'cl_cm_asa.png')
TEX_Q2 = os.path.join(TEXDIR, 'item2_resultados.tex')
for p in (DESIGNTOOL_DIR, LAB04,
          os.path.join(DESIGNTOOL_DIR, 'lab02_opt', 'otimizacao_NJ0502')):
    if p not in sys.path:
        sys.path.insert(0, p)

import avl_tools as at                                     # noqa: E402
import opt_common as oc                                    # noqa: E402
from designTool.constants import gravity                   # noqa: E402


def _qhat(n_z, V_tas, cref):
    '''qc/2V no AVL: q = g(n-1)/V, adimensionalizado com Cref.'''
    q = gravity*(n_z - 1.0)/V_tas
    return q, q*cref/(2.0*V_tas)


def _lift_surface(fs, nome, qdyn):
    '''Sustentacao [N]. Com YDUPLICATE o fs so traz a metade direita.'''
    d = fs[nome]
    L = float(qdyn*np.sum(d['cl']*d['Area']))
    if np.min(d['Yle']) >= -1e-6:
        L *= 2.0
    return L


def _pesos(ap, W_N):
    ew = ap['empty_weight']
    n_eng = ap['inputs']['n_engines']
    W_empty = ew['W_empty']
    W_pay = ap['inputs']['W_payload']
    W_crew = ap['inputs']['W_crew']
    W_fuel = W_N - W_empty - W_pay - W_crew
    return {
        'W_N': float(W_N),
        'W_kgf': float(W_N/gravity),
        'W_asa_N': float(ew['W_w']),
        'W_asa_kgf': float(ew['W_w']/gravity),
        'W_fuel_N': float(W_fuel),
        'W_fuel_kgf': float(W_fuel/gravity),
        'W_mlg_cada_N': float(ew['W_mlg']/2.0),
        'W_mlg_cada_kgf': float(ew['W_mlg']/(2.0*gravity)),
        'W_motor_cada_N': float(ew['W_eng']/n_eng),
        'W_motor_cada_kgf': float(ew['W_eng']/(n_eng*gravity)),
        'W_empty_kgf': float(W_empty/gravity),
    }


def _caso_avl(nome, cond, base, it, avl):
    n_z = float(cond['n_max'])
    V_eas = float(cond['velocidades_EAS']['VA'])
    V_tas = float(cond['velocidades_TAS']['VA'])
    Mach = float(cond['mach']['VA'])
    CLmax = float(base['CLmax'])
    cref = float(base['cref'])
    Sref = float(base['Sref'])
    rho = float(base['rho'])
    q, qhat = _qhat(n_z, V_tas, cref)
    qdyn = 0.5*rho*V_tas**2
    CL_nec = n_z*cond['W_N']/(qdyn*Sref)

    r = at.run_cases(avl, Mach, [{
        'CL': CLmax, 'trim': True, 'qc2V': qhat,
    }], it=it, tag='q2_%s' % nome)[0]

    fs = r['fs']
    wing = at.wing_strips(fs, 'Wing')
    L_asa = _lift_surface(fs, 'Wing', qdyn)
    L_eh = _lift_surface(fs, 'Stab', qdyn)
    eta = 2.0*wing['Yle']/base['bref']

    return {
        'nome': nome,
        'avl': avl,
        'it_deg': it,
        'n_z': n_z,
        'W_N': float(cond['W_N']),
        'V_EAS': V_eas,
        'V_TAS': V_tas,
        'Mach': Mach,
        'q_rad_s': q,
        'qc2V': qhat,
        'qdyn_Pa': qdyn,
        'CL_necessario': float(CL_nec),
        'CL_alvo': CLmax,
        'Alpha_deg': r['Alpha'],
        'CLtot': r['CLtot'],
        'Cmtot': r['Cmtot'],
        'CDtot': r['CDtot'],
        'elevator_deg': r['elevator'],
        'qc2V_avl': r['qc2V'],
        'L_asa_N': L_asa,
        'L_eh_N': L_eh,
        'L_total_strips_N': L_asa + L_eh,
        'L_CLtot_N': float(r['CLtot']*qdyn*Sref),
        'L_nW': n_z*cond['W_N'],
        'eta': eta.tolist(),
        'Yle_m': wing['Yle'].tolist(),
        'cl': wing['cl'].tolist(),
        'cm_c4': wing['cm_c4'].tolist(),
        'c_m': wing['Chord'].tolist(),
    }


def _plot(casos, path):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    estilos = {'inicio': ('#1f4e79', '-'), 'fim': ('#c53030', '--')}
    rotulo = {'inicio': 'início do cruzeiro', 'fim': 'fim do cruzeiro'}
    for c in casos:
        cor, ls = estilos[c['nome']]
        lab = rotulo[c['nome']]
        axes[0].plot(c['eta'], c['cl'], ls, color=cor, lw=1.8, label=lab)
        axes[1].plot(c['eta'], c['cm_c4'], ls, color=cor, lw=1.8, label=lab)
    axes[0].set_ylabel(r'$c_\ell$')
    axes[0].set_title(r'Distribuição na asa em $V_A$, $n_z = 2{,}5$ (CG dianteiro)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(framealpha=0.92)
    axes[1].set_ylabel(r'$c_m$ no $c/4$')
    axes[1].set_xlabel(r'$\eta = 2y/b$')
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _csv_strips(caso, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['eta', 'Yle_m', 'c_m', 'cl', 'cm_c4'])
        for row in zip(caso['eta'], caso['Yle_m'], caso['c_m'],
                       caso['cl'], caso['cm_c4']):
            w.writerow(['%.6f' % v for v in row])


def _milhar(x, nd=0):
    s = '%.*f' % (nd, x)
    inteiro, _, frac = s.partition('.')
    g = []
    while inteiro:
        g.append(inteiro[-3:])
        inteiro = inteiro[:-3]
    return r'\,'.join(reversed(g)) + ('{,}' + frac if frac else '')


def _kgf_N(kgf, N):
    return r'$%s$ kgf ($%s$ N)' % (_milhar(kgf), _milhar(N))


def _escreve_relatorio(saida, path=TEX_Q2):
    '''Secao 2 do relatorio: so o que o roteiro pede.'''
    ci, cf = saida['condicoes']['inicio'], saida['condicoes']['fim']
    pi, pf = ci['pesos'], cf['pesos']
    ai, af = ci['avl'], cf['avl']
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def par(rot, vi, vf):
        return '%s & %s & %s \\\\' % (rot, vi, vf)

    linhas = [
        r'% NJ-0502 --- Atividade 06, secao 2 do relatorio',
        r'% \input{item2_resultados}  (depois da secao 1)',
        r'% Copie tambem cl_cm_asa.png para a mesma pasta no Overleaf.',
        r'',
        r'\section{Simulações Aerodinâmicas}',
        r'\label{sec:simulacoes-aero}',
        r'',
        r'As simulações foram feitas no AVL para o início e o fim do '
        r'cruzeiro, no ponto de manobra $V_A$ com fator de carga máximo '
        r'($n_z = 2{,}5$). A aeronave está trimada e a taxa de arfagem '
        r'$q = g(n_z-1)/V_{\mathrm{TAS}}$ foi prescrita.',
        r'',
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Pesos na condição de manobra ($V_A$, $n_{\max}$).}',
        r'\label{tab:q2-pesos}',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r' & Início do cruzeiro & Fim do cruzeiro \\',
        r'\midrule',
        par(r'Peso da aeronave',
            _kgf_N(pi['W_kgf'], pi['W_N']),
            _kgf_N(pf['W_kgf'], pf['W_N'])),
        par(r'Peso da asa',
            _kgf_N(pi['W_asa_kgf'], pi['W_asa_N']),
            _kgf_N(pf['W_asa_kgf'], pf['W_asa_N'])),
        par(r'Peso do combustível',
            _kgf_N(pi['W_fuel_kgf'], pi['W_fuel_N']),
            _kgf_N(pf['W_fuel_kgf'], pf['W_fuel_N'])),
        par(r'Peso de cada MLG',
            _kgf_N(pi['W_mlg_cada_kgf'], pi['W_mlg_cada_N']),
            _kgf_N(pf['W_mlg_cada_kgf'], pf['W_mlg_cada_N'])),
        par(r'Peso de cada motor (instalado)',
            _kgf_N(pi['W_motor_cada_kgf'], pi['W_motor_cada_N']),
            _kgf_N(pf['W_motor_cada_kgf'], pf['W_motor_cada_N'])),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        r'',
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Forças de sustentação em $V_A$, $n_z = 2{,}5$.}',
        r'\label{tab:q2-forcas}',
        r'\begin{tabular}{lrr}',
        r'\toprule',
        r' & Início do cruzeiro & Fim do cruzeiro \\',
        r'\midrule',
        par(r'Sustentação da asa',
            _kgf_N(ai['L_asa_N']/gravity, ai['L_asa_N']),
            _kgf_N(af['L_asa_N']/gravity, af['L_asa_N'])),
        par(r'Sustentação da EH',
            _kgf_N(ai['L_eh_N']/gravity, ai['L_eh_N']),
            _kgf_N(af['L_eh_N']/gravity, af['L_eh_N'])),
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
        r'',
        r'\begin{figure}[htbp]',
        r'\centering',
        r'\includegraphics[width=0.92\textwidth]{cl_cm_asa.png}',
        r'\caption{Distribuição de $c_\ell$ e de $c_m$ no $c/4$ ao longo da '
        r'asa em $V_A$, $n_z = 2{,}5$.}',
        r'\label{fig:q2-cl-cm}',
        r'\end{figure}',
        r'',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n')
    fig_dest = os.path.join(os.path.dirname(path), 'cl_cm_asa.png')
    if os.path.exists(FIG_Q2) and os.path.abspath(FIG_Q2) != os.path.abspath(fig_dest):
        shutil.copy2(FIG_Q2, fig_dest)


def _imprime(pesos, aero):
    print('\n=== %s do cruzeiro  (fwd.avl, trimado, q prescrito) ===' % aero['nome'])
    print('  Peso aeronave     %10.0f N   (%8.0f kgf)'
          % (pesos['W_N'], pesos['W_kgf']))
    print('  Peso da asa       %10.0f N   (%8.0f kgf)'
          % (pesos['W_asa_N'], pesos['W_asa_kgf']))
    print('  Peso combustivel  %10.0f N   (%8.0f kgf)'
          % (pesos['W_fuel_N'], pesos['W_fuel_kgf']))
    print('  Cada MLG          %10.0f N   (%8.0f kgf)'
          % (pesos['W_mlg_cada_N'], pesos['W_mlg_cada_kgf']))
    print('  Cada motor (inst) %10.0f N   (%8.0f kgf)'
          % (pesos['W_motor_cada_N'], pesos['W_motor_cada_kgf']))
    print('  V_EAS / V_TAS     %7.2f / %7.2f m/s   Mach %.3f'
          % (aero['V_EAS'], aero['V_TAS'], aero['Mach']))
    print('  CL alvo (CLmax DT) %.4f   CL AVL %.4f   alpha %.3f deg'
          % (aero['CL_alvo'], aero['CLtot'], aero['Alpha_deg']))
    print('  CM                 %.5f   elevator %+.3f deg'
          % (aero['Cmtot'], aero['elevator_deg']))
    print('  q                  %.5f rad/s   qc/2V pedido %.6f  AVL %.6f'
          % (aero['q_rad_s'], aero['qc2V'], aero['qc2V_avl']))
    print('  L asa / L EH       %.0f / %+.0f N'
          % (aero['L_asa_N'], aero['L_eh_N']))
    print('  n W / CL q S       %.0f / %.0f N'
          % (aero['L_nW'], aero['L_CLtot_N']))


def main():
    if '--tex' in sys.argv:
        saida = json.load(open(JSON_Q2, encoding='utf-8'))
        _escreve_relatorio(saida)
        print('gravado', TEX_Q2)
        return 0

    env = json.load(open(JSON_Q1, encoding='utf-8'))
    base = env['base']
    if abs(base['CLmax'] - base['CLmax_DT']) > 1e-6:
        raise SystemExit('o JSON do item 1 nao esta com CLmax do designTool')

    ap = oc.run_designTool(oc.get_baseline())
    it = float(base['it_deg'])
    os.makedirs(RESDIR, exist_ok=True)

    saida = {'base': {
        'CLmax': base['CLmax'], 'CLmax_fonte': base['CLmax_fonte'],
        'n_max': 2.5, 'h_ft': base['h_ft'], 'Sref': base['Sref'],
        'cref': base['cref'], 'it_deg': it, 'avl': 'fwd.avl',
        'washout_deg': base['washout_deg'],
    }, 'condicoes': {}}

    casos_plot = []
    for nome in ('inicio', 'fim'):
        cond = env['condicoes'][nome]
        pesos = _pesos(ap, cond['W_N'])
        aero = _caso_avl(nome, cond, base, it, 'fwd.avl')
        _imprime(pesos, aero)
        casos_plot.append(aero)
        _csv_strips(aero, os.path.join(RESDIR, 'strips_%s.csv' % nome))
        saida['condicoes'][nome] = {'pesos': pesos, 'avl': {
            k: v for k, v in aero.items()
            if k not in ('eta', 'Yle_m', 'cl', 'cm_c4', 'c_m')
        }}

    _plot(casos_plot, FIG_Q2)
    with open(JSON_Q2, 'w', encoding='utf-8') as f:
        json.dump(saida, f, indent=2)
    _escreve_relatorio(saida)
    print('\ngravado em', RESDIR)
    print('relatorio em', TEX_Q2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
