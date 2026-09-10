'''
PRJ-23 Lab 03 - Busca o alpha do NACA 1411 que atinge cl_ref no Euler.
Uso (a partir desta pasta): python find_alpha.py
'''

import os
import sys
import csv
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
eulerblock_path = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, eulerblock_path)

from eulerblock import euler_mod as eb

# Ponto de projeto (meio do cruzeiro, Lab 03)
CL_REF = 0.470976947932197
MACH = 0.85

# NACA 1411 (CST)
Al = [-0.1489439, -0.10330027, -0.10305128, -0.10514982]
Au = [0.16146332, 0.18349204, 0.14126241, 0.18194397]

# Mesma malha/solver do single_run.py (sem adjoint/plot, para acelerar a busca)
order = 2
iter_max = 20000
dt = 0.001
CFL = 0.2
use_local_dt = 1
res_NK = 1e-4
res_tol = 1e-6
grid_level = 1.0
Nchord0, NJ0, s00 = 30, 48, 0.5e-2
Nchord = int((Nchord0*grid_level) + 1)
NJ = int((NJ0*grid_level) + 1)
s0 = NJ0/(NJ - 1)*s00

RESULTS_DIR = os.path.abspath(os.path.join(
    _HERE, '..', '..', '..', '..', 'resultados_eulerblock'))


def run_cl(alpha_rad, reinitialize=0):
    t0 = time.time()
    results = eb.run_cst(
        Al, Au, Nchord, alpha_rad, MACH, 1.4, order,
        iter_max, dt, CFL, use_local_dt, res_NK, res_tol,
        reinitialize, adj_funcs=[], plot=False, NJ=NJ, s0=s0)
    elapsed = time.time() - t0
    print('  alpha=%.4f deg  CL=%.6f  CD=%.6f  (%.1f s)'
          % (alpha_rad*180/np.pi, results['CL'], results['CD'], elapsed))
    return results


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.chdir(_HERE)

    # Partidas em graus (em torno do valor tipico do exemplo)
    alphas_deg = [1.0, 1.4, 1.8]
    hist = []
    for i, a_deg in enumerate(alphas_deg):
        res = run_cl(a_deg*np.pi/180.0, reinitialize=(0 if i else 1))
        hist.append({
            'alpha_deg': a_deg,
            'CL': float(res['CL']),
            'CD': float(res['CD']),
            'CM': float(res['CM']),
        })

    a = np.array([h['alpha_deg'] for h in hist])
    cl = np.array([h['CL'] for h in hist])
    # Interpolacao linear local ao redor do alvo
    alpha_star = float(np.interp(CL_REF, cl, a))
    # Se estiver fora da faixa amostrada, extrapola com os dois mais proximos
    if CL_REF < cl.min() or CL_REF > cl.max():
        # ajuste linear global
        p = np.polyfit(cl, a, 1)
        alpha_star = float(np.polyval(p, CL_REF))

    print('  alpha interpolado = %.4f deg para CL_ref = %.4f' % (alpha_star, CL_REF))

    res_f = run_cl(alpha_star*np.pi/180.0, reinitialize=0)
    hist.append({
        'alpha_deg': alpha_star,
        'CL': float(res_f['CL']),
        'CD': float(res_f['CD']),
        'CM': float(res_f['CM']),
    })

    # Secante com um passo extra se ainda estiver longe
    err = res_f['CL'] - CL_REF
    if abs(err) > 5e-4:
        # usa o ultimo e o ponto de hist mais proximo
        j = int(np.argmin(np.abs(cl - CL_REF)))
        a0, cl0 = hist[j]['alpha_deg'], hist[j]['CL']
        a1, cl1 = alpha_star, res_f['CL']
        if abs(cl1 - cl0) > 1e-8:
            alpha_star2 = a1 - (cl1 - CL_REF)*(a1 - a0)/(cl1 - cl0)
            print('  correcao secante -> %.4f deg' % alpha_star2)
            res_f = run_cl(alpha_star2*np.pi/180.0, reinitialize=0)
            alpha_star = alpha_star2
            hist.append({
                'alpha_deg': alpha_star,
                'CL': float(res_f['CL']),
                'CD': float(res_f['CD']),
                'CM': float(res_f['CM']),
            })

    csv_path = os.path.join(RESULTS_DIR, 'alpha_naca1411.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['alpha_deg', 'CL', 'CD', 'CM'])
        w.writeheader()
        for row in hist:
            w.writerow(row)
        w.writerow({
            'alpha_deg': alpha_star,
            'CL': float(res_f['CL']),
            'CD': float(res_f['CD']),
            'CM': float(res_f['CM']),
        })

    summary = os.path.join(RESULTS_DIR, 'alpha_escolhido.csv')
    with open(summary, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['parametro', 'valor'])
        w.writerow(['mach', MACH])
        w.writerow(['CL_ref', CL_REF])
        w.writerow(['alpha_deg', alpha_star])
        w.writerow(['alpha_rad', alpha_star*np.pi/180.0])
        w.writerow(['CL', float(res_f['CL'])])
        w.writerow(['CD', float(res_f['CD'])])
        w.writerow(['CM', float(res_f['CM'])])
        w.writerow(['erro_CL', float(res_f['CL']) - CL_REF])

    print('  gravado:', csv_path)
    print('  gravado:', summary)
    print('  RESULTADO: alpha = %.4f deg  CL = %.6f  (alvo %.4f)'
          % (alpha_star, res_f['CL'], CL_REF))
