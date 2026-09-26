'''
PRJ-23 Lab 04 - Ponto de projeto, condicao de baixa velocidade e polar do
designTool, compartilhados pelos itens 4 e 5.

O ponto de projeto e o MESMO do item 3 (q3_incidencia_eh.py) e do Lab 03: meio
do cruzeiro com o my_airplane vigente, calculado por
lab03_perfil/airfoil_mod/run_1_ponto_projeto.py. O enunciado permite
explicitamente essa escolha ("ou para o mesmo ponto de projeto usado no
Lab 03") e ela mantem o relatorio coerente - o perfil do Lab 03 foi otimizado
nesse cl e o it dos arquivos .avl foi ajustado nesse CL.

Tambem devolve:
  - a incidencia de empenagem it de cada CG, lida de resultados_q3/;
  - o CD0 de cruzeiro do designTool (o valor gravado no campo "# CDp" dos
    arquivos fwd.avl/aft.avl);
  - a polar completa do designTool, CD(CL), para comparacao com o AVL no
    item 5;
  - a condicao de baixa velocidade (V2 de decolagem) usada no item 4.

Uso: python ponto_projeto.py   (grava resultados_comum/ponto_projeto.json)
'''

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGNTOOL_DIR = os.path.abspath(os.path.join(HERE, '..'))
OPTDIR = os.path.join(DESIGNTOOL_DIR, 'lab02_opt', 'otimizacao_NJ0502')
AIRFOIL_MOD = os.path.join(DESIGNTOOL_DIR, 'lab03_perfil', 'airfoil_mod')
RESDIR = os.path.join(HERE, 'resultados_comum')
Q3_CSV = os.path.join(HERE, 'resultados_q3', 'q3_incidencia_eh.csv')
for p in (DESIGNTOOL_DIR, OPTDIR, AIRFOIL_MOD):
    if p not in sys.path:
        sys.path.insert(0, p)

import opt_common as oc                                    # noqa: E402
import run_1_ponto_projeto as rpp                          # noqa: E402
from designTool.aerodynamics import aerodynamics           # noqa: E402
from designTool.auxiliary import atmosphere                # noqa: E402
from designTool.constants import gravity                   # noqa: E402


def angulo_asa(arquivo='fwd.avl'):
    '''Le o cartao ANGLE da superficie Wing do .avl (incidencia da asa).'''
    caminho = os.path.join(DESIGNTOOL_DIR, 'AVL_package', arquivo)
    linhas = open(caminho, encoding='utf-8',
                  errors='replace').read().splitlines()
    for k, ln in enumerate(linhas):
        if ln.strip().upper() == 'ANGLE':
            return float(linhas[k + 1].split()[0])
    return 0.0


def ponto_projeto(ap=None):
    '''Ponto de projeto do Lab 03 / item 3, com os dados extras do Lab 04.'''
    ap = oc.run_designTool(oc.get_baseline()) if ap is None else ap
    I, G, B, T = ap['inputs'], ap['geometry'], ap['balance'], ap['thrust_matching']

    pp3 = rpp.run()                 # mesma rotina usada pelo item 3
    M, CL = pp3['M'], pp3['clref']
    h = I['altitude_cruise']
    atm = atmosphere(h)
    rho, a, mu = atm['density'], atm['speed_of_sound'], atm['dyn_viscosity']
    V = M*a

    _, _, dd = aerodynamics(ap, M, h, CL, n_engines_failed=0,
                            highlift_config='clean', lg_down=0, h_ground=0)

    return {
        'fonte': 'lab03_perfil/airfoil_mod/run_1_ponto_projeto.py '
                 '(meio do cruzeiro), o mesmo do item 3',
        'W0_N': float(T['W0']), 'W0_kgf': float(pp3['W0_kgf']),
        'W_N': float(pp3['W_kgf']*gravity), 'W_kgf': float(pp3['W_kgf']),
        'W_ini_kgf': float(pp3['W_ini_kgf']),
        'W_fim_kgf': float(pp3['W_fim_kgf']),
        'h_m': float(h), 'h_ft': float(pp3['h_ft']),
        'rho': float(rho), 'a': float(a), 'mu': float(mu),
        'M': float(M), 'V': float(V), 'q': float(0.5*rho*V**2),
        'CL': float(CL),
        'Sref': float(I['S_w']), 'cref': float(G['cm_w']),
        'bref': float(G['b_w']),
        'CD0_cruzeiro': float(dd['CD0']),
        'CD_DT_no_ponto': float(dd['CD']),
        'CDind_DT_no_ponto': float(dd['CDind']),
        'CDwave_DT_no_ponto': float(dd['CDwave']),
        'K_DT': float(dd['K']),
        'CLmax_clean_DT': float(dd['CLmax_clean']),
        'xcg_fwd': float(B['xcg_fwd']), 'xcg_aft': float(B['xcg_aft']),
        'xnp_DT': float(B['xnp']),
        'SM_fwd_DT': float(B['SM_fwd']), 'SM_aft_DT': float(B['SM_aft']),
        'clmax_w_DT': float(I['clmax_w']),
        'iw_deg': float(angulo_asa()),
        'Re_cref': float(rho*V*G['cm_w']/mu),
    }


def incidencias_eh():
    '''it de cada CG, do item 3 (resultados_q3/q3_incidencia_eh.csv).'''
    with open(Q3_CSV, encoding='utf-8') as f:
        linhas = list(csv.DictReader(f))
    return {r['caso']: {'arquivo': r['arquivo'],
                        'it_deg': float(r['it_deg']),
                        'delta_e_deg': float(r['delta_e_deg']),
                        'alpha_deg': float(r['alpha_deg']),
                        'CL': float(r['CLtot']), 'CD': float(r['CDtot'])}
            for r in linhas}


def polar_designTool(ap, pp, CLs):
    '''CD(CL) do designTool no Mach/altitude do ponto de projeto, asa limpa.'''
    CD = []
    for CL in np.atleast_1d(CLs):
        _, _, dd = aerodynamics(ap, pp['M'], pp['h_m'], float(CL),
                                n_engines_failed=0, highlift_config='clean',
                                lg_down=0, h_ground=0)
        CD.append(float(dd['CD']))
    return np.array(CD)


def condicao_baixa_velocidade(ap):
    '''Condicao de baixa velocidade usada no metodo da secao critica.

    Adota-se V2 = 1,2 Vs no nivel do mar, com W0 e o CLmax de decolagem do
    designTool - exatamente a condicao (Re, M) em que o clmax do perfil
    otimizado foi levantado no XFOIL durante o Lab 03. Assim o limite de
    cl_max usado aqui e o mesmo daquele relatorio.
    '''
    I, G = ap['inputs'], ap['geometry']
    atm = atmosphere(I['altitude_takeoff'])
    rho, a, mu = atm['density'], atm['speed_of_sound'], atm['dyn_viscosity']
    W0 = ap['thrust_matching']['W0']
    CLmaxTO = ap['thrust_matching']['CLmaxTO']
    Vs = np.sqrt(2*W0/(rho*I['S_w']*CLmaxTO))
    V = 1.2*Vs
    return {'h_m': float(I['altitude_takeoff']), 'rho': float(rho),
            'a': float(a), 'mu': float(mu),
            'CLmaxTO_DT': float(CLmaxTO),
            'clmax_w_DT': float(I['clmax_w']),
            'Vs_TO': float(Vs), 'V2': float(V),
            'M': float(V/a), 'Re_cref': float(rho*V*G['cm_w']/mu),
            'Re_por_metro': float(rho*V/mu)}


def main():
    os.makedirs(RESDIR, exist_ok=True)
    ap = oc.run_designTool(oc.get_baseline())
    pp = ponto_projeto(ap)
    bv = condicao_baixa_velocidade(ap)
    its = incidencias_eh()
    with open(os.path.join(RESDIR, 'ponto_projeto.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'ponto_projeto': pp, 'baixa_velocidade': bv,
                   'incidencias_eh': its}, f, indent=2)
    print('Ponto de projeto (meio do cruzeiro, igual ao item 3 e ao Lab 03):')
    for k in ('W0_N', 'W_N', 'h_m', 'rho', 'a', 'M', 'V', 'CL', 'Sref',
              'cref', 'bref', 'iw_deg', 'CD0_cruzeiro', 'CD_DT_no_ponto',
              'CLmax_clean_DT', 'xcg_fwd', 'xcg_aft', 'Re_cref'):
        print('  %-16s %14.6f' % (k, pp[k]))
    print('Incidencias de empenagem (item 3):')
    for tag, d in its.items():
        print('  %-4s it = %+8.4f deg  (delta_e = %+.4f, alpha = %+.4f)'
              % (tag, d['it_deg'], d['delta_e_deg'], d['alpha_deg']))
    print('Baixa velocidade (V2 de decolagem, nivel do mar):')
    for k in ('CLmaxTO_DT', 'Vs_TO', 'V2', 'M', 'Re_cref', 'Re_por_metro'):
        print('  %-16s %14.6f' % (k, bv[k]))
    print('\ngravado em', os.path.join(RESDIR, 'ponto_projeto.json'))


if __name__ == '__main__':
    main()
