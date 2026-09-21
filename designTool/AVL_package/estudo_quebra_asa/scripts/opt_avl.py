# -*- coding: utf-8 -*-
"""
Otimizacao da NJ-0502 com margem estatica corrigida pelo AVL.

Dentro do SLSQP (barato, analitico):
  - quebra (BF interno vertical) que leva o trem a CRANK_XI da corda local
  - correcoes da quebra: CG da asa e do combustivel, MAC, volume de tanque,
    deslocamento do quarto de corda do MAC (k * d_qc)
  - restricoes de estabilidade/trem/tanque reavaliadas com essas correcoes
Fora do SLSQP (AVL, modelo completo):
  - delta = Xnp_AVL - (Xnp_designTool + k*d_qc), atualizado ate convergir
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

import opt_nac as on
import opt_common as oc
import avl_check as ac

K_QC = 0.85            # d(Xnp)/d(quarto de corda do MAC), calibrado no AVL
STATE = {'xi': None, 'delta': 0.0}
FUEL_CREDIT = False   # True: tanque ocupa 20-60% da corda ESTENDIDA pela quebra
DV13 = ['S_w', 'AR_w', 'sweep_w', 'xr_w', 'Cht', 'Lc_h', 'Cvt', 'Lb_v',
        'x_mlg', 'y_mlg', 'z_lg', 'x_n', 'y_n']


def _planform(r, yb):
    L, t, ct, xr = r['semi_b'], r['tLE'], r['ct_w'], r['xr_w']
    S2 = r['S_w']/2
    A = yb + (L - yb)/2.
    B = -t*yb*yb/2. + (L - yb)*(-t*yb + ct)/2.
    CR = (S2 - B)/A
    CB = CR - t*yb
    y = np.linspace(0, L, 3001)
    c = np.where(y <= yb, CR - t*y, CB + (ct - CB)*(y - yb)/max(L - yb, 1e-9))
    a = np.trapezoid(c, y)
    mac = np.trapezoid(c*c, y)/a
    xm = np.trapezoid(c*(xr + t*y), y)/a
    m = y <= 0.98*L
    tc = r['tcr_w'] + (r['tct_w'] - r['tcr_w'])*y/L
    wg = (c**2*tc)[m]
    xf = (xr + t*y + (r['x_tank'] + 0.5*r['c_tank'])*c)[m]
    vol = np.trapezoid(wg, y[m])
    return dict(MAC=mac, qc=xm + mac/4, xcg_w=xm + 0.4*mac,
                xcg_f=np.trapezoid(wg*xf, y[m])/vol, vol=vol)


_extract1 = oc.extract


def extract(ap):
    r = _extract1(ap)
    I, B, E, T = ap['inputs'], ap['balance'], ap['empty_weight'], ap['thrust_matching']
    r.update(tcr_w=I['tcr_w'], tct_w=I['tct_w'], x_tank=I['x_tank_c_w'],
             c_tank=I['c_tank_c_w'], xnp_dt=B['xnp'], xcg_aft_dt=B['xcg_aft'],
             xcg_fwd_dt=B['xcg_fwd'], x_nlg=I['x_nlg'])
    P0 = _planform(r, 1e-9)
    yb = 1e-9
    if STATE['xi'] is not None:
        c = on.crank(r, STATE['xi'])
        if c['c_root'] > r['cr_w']:
            yb = c['y_b']
    r['yb_used'] = yb
    P = _planform(r, max(yb, 1e-9)) if yb > 1e-6 else P0
    dcg = (E['W_w']*(P['xcg_w'] - P0['xcg_w'])
           + (T['W_fuel']*(P['xcg_f'] - P0['xcg_f']) if FUEL_CREDIT else 0.0))/T['W0']
    xa, xf = B['xcg_aft'] + dcg, B['xcg_fwd'] + dcg
    xnp = B['xnp'] + K_QC*(P['qc'] - P0['qc']) + STATE['delta']
    r['dcg'] = dcg
    r['MAC_c'] = P['MAC']
    r['xnp_c'] = xnp
    r['SM_aft'] = (xnp - xa)/P['MAC']
    r['SM_fwd'] = (xnp - xf)/P['MAC']
    r['alpha_tipback'] = np.arctan((I['x_mlg'] - xa)/(-I['z_lg']))
    r['frac_nlg_aft'] = (I['x_mlg'] - xa)/(I['x_mlg'] - I['x_nlg'])
    r['frac_nlg_fwd'] = (I['x_mlg'] - xf)/(I['x_mlg'] - I['x_nlg'])
    if FUEL_CREDIT:
        r['tank_excess'] = (1 + B['tank_excess'])*P['vol']/P0['vol'] - 1
    return r


oc.extract = extract
CRANK = ['crank_cb', 'crank_ybmax', 'crank_ybmin']
BASE = [c for c in oc.CON_NAMES if c not in CRANK]


def optimize(inputs, xi, maxiter=300):
    STATE['xi'] = xi
    if xi is not None:
        on.CRANK_XI = xi
    cons = BASE + (CRANK if xi is not None else [])
    m = on.Model(DV13, baseline_inputs=inputs, con_names=cons)
    res = minimize(m.objfun, m.x0, jac=m.objgrad,
                   constraints=[{'type': 'ineq', 'fun': m.confun, 'jac': m.conjac}],
                   bounds=m.bounds, method='slsqp',
                   options={'maxiter': maxiter, 'ftol': 1e-7, 'disp': False})
    return m.build_inputs(res.x), m.results(res.x), cons


def avl_delta(inputs, r):
    """Xnp do AVL no ponto atual e o novo delta."""
    e = ac.evaluate(inputs, r['yb_used'] if r['yb_used'] > 1e-6 else 1e-6,
                    "opt_tmp.avl")
    pred_wo_delta = r['xnp_c'] - STATE['delta']
    return e, e['Xnp'] - pred_wo_delta


def solve_corrected(inputs, xi, delta0=0.0, tol=0.01, nmax=8, verbose=True):
    STATE['delta'] = delta0
    inp = inputs
    for k in range(nmax):
        inp, r, cons = optimize(inp, xi)
        e, dnew = avl_delta(inp, r)
        if verbose:
            print("    it %d: delta %+.3f -> %+.3f m | W0 %.0f kgf | SM_aft AVL %.2f%% | xi %.3f"
                  % (k, STATE['delta'], dnew, r['W0']/oc.gravity, 100*e['SM_aft'], e['xi']))
        if abs(dnew - STATE['delta']) < tol:
            STATE['delta'] = dnew
            inp, r, cons = optimize(inp, xi)
            e, _ = avl_delta(inp, r)
            break
        STATE['delta'] = dnew
    g = oc.constraint_vector(r, cons)
    return inp, r, e, g, cons
