# -*- coding: utf-8 -*-
"""Projeto final: otimo com xi=0.70 e painel externo afilado (c_b >= 2 c_t)."""
import json
import os as _os
import sys as _s
_s.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from avl_check import RESDIR
import os
import numpy as np
import opt_avl as oa
import opt_nac as on
import opt_common as oc
import avl_check as ac
from designTool.geometry import control_surface_area_fraction

oa.FUEL_CREDIT = False
on.CRANK_CB_CT = 2.0
XI = 0.70
rows = json.load(open(_os.path.join(RESDIR, 'avl_opt_rows_cons_cb2.json')))
fin = [r for r in rows if r['tag'] == 'q0.70'][0]
inp = fin['inputs']
oa.STATE.update(xi=XI, delta=fin['delta']); on.CRANK_XI = XI

base = oc.get_baseline()
oa.STATE.update(xi=None, delta=0.0)
ap0 = oc.run_designTool(base); r0 = oc.extract(ap0)
e0 = ac.evaluate(base, 1e-6, 'tmp0.avl')
oa.STATE.update(xi=XI, delta=fin['delta'])
ap = oc.run_designTool(inp); r = oc.extract(ap)
c = on.crank(r, XI)
I, G = ap['inputs'], ap['geometry']
L = G['b_w']/2
P = ac.planform(ap, c['y_b'])
ch = P['ch']
ysob = I['D_f']/2
cf = I['c_flap_c_wing']

# ---------------- FLAP: area EXPOSTA (fora da fuselagem) constante ----------------
Dfb = I['D_f']/G['b_w']
S_exp0 = control_surface_area_fraction(cf, Dfb, I['b_flap_b_wing'], I['taper_w'])*I['S_w']/2
y = lambda a, b: np.linspace(a, b, 20001)


def area_cfc(yf):             # c_f/c constante
    g = y(ysob, yf); return cf*np.trapezoid([ch(v) for v in g], g)


def area_cc(yf):              # corda de flap constante (= cf*c_b) por dentro da quebra
    yb = c['y_b']
    a_in = cf*c['c_b']*(min(yf, yb) - ysob)
    if yf <= yb:
        return a_in
    g = y(yb, yf); return a_in + cf*np.trapezoid([ch(v) for v in g], g)


def solve(fun):
    lo, hi = ysob + 0.1, L
    if fun(hi) < S_exp0:
        return float('nan')
    for _ in range(200):
        mid = (lo + hi)/2
        if fun(mid) < S_exp0: lo = mid
        else: hi = mid
    return (lo + hi)/2


yf_cfc, yf_cc = solve(area_cfc), solve(area_cc)
ya = (1 - I['b_ail_b_wing'])*L
print("FLAP (area exposta por semi-asa mantida = %.2f m2)" % S_exp0)
print("  trapezio (Lab 02 conv.)  : ate %.2f m (%.1f%% b/2)  gap p/ aileron %.2f m" % (I['b_flap_b_wing']*L, 100*I['b_flap_b_wing'], ya - I['b_flap_b_wing']*L))
print("  quebra, c_f/c = %.2f     : ate %.2f m (%.1f%% b/2)  gap p/ aileron %.2f m" % (cf, yf_cfc, 100*yf_cfc/L, ya - yf_cfc))
print("  quebra, c_f const. interno: %s" % ("inviavel (nao recupera a area)" if np.isnan(yf_cc) else "ate %.2f m (%.1f%% b/2)" % (yf_cc, 100*yf_cc/L)))
print("  corda do flap: raiz exposta %.2f m | quebra %.2f m" % (cf*ch(ysob), cf*c['c_b']))
# enflechamento da articulacao
t = P['t']
hin_in = np.degrees(np.arctan(0.32*t))
sl = (P['ct'] - c['c_b'])/(L - c['y_b'])
hin_out = np.degrees(np.arctan(t + 0.68*sl))
from designTool.geometry import change_sweep
hin0 = np.degrees(change_sweep(0.25, 1 - cf, I['sweep_w'], L, G['cr_w'], G['ct_w']))
print("  articulacao: trapezio %.1f deg -> interno %.1f / externo %.1f deg" % (hin0, hin_in, hin_out))

# ---------------- relatorio ----------------
e = ac.evaluate(inp, c['y_b'], 'tmp_relatorio.avl')
g = oc.constraint_vector(r, oa.BASE + oa.CRANK)
print()
print("RESTRICOES ativas: " + ", ".join(n for n, v in zip(oa.BASE + oa.CRANK, g) if abs(v) < 2e-3))
print("RESTRICOES violadas: " + (", ".join(n for n, v in zip(oa.BASE + oa.CRANK, g) if v < -2e-3) or "nenhuma"))
T, T0 = ap['thrust_matching'], ap0['thrust_matching']
dv = ['S_w', 'AR_w', 'sweep_w', 'xr_w', 'Cht', 'Lc_h', 'Cvt', 'Lb_v', 'x_mlg', 'y_mlg', 'z_lg', 'x_n', 'y_n', 'z_n']
print()
print("%-10s %12s %12s" % ("var", "Lab 02", "novo"))
for n in dv:
    f = oc.rad2deg if n == 'sweep_w' else 1.0
    print("%-10s %12.4f %12.4f" % (n, base[n]*f, inp[n]*f))
print()
for lab, a, b in [("W0 [kgf]", T0['W0']/oc.gravity, T['W0']/oc.gravity),
                  ("W_fuel [kgf]", T0['W_fuel']/oc.gravity, T['W_fuel']/oc.gravity),
                  ("b_w [m]", G['b_w']*0 + r0['b_w'], r['b_w']),
                  ("c_raiz trap. [m]", r0['cr_w'], r['cr_w']),
                  ("c_raiz quebra [m]", r0['cr_w'], c['c_root']),
                  ("x BF vertical [m]", np.nan, c['x_te']),
                  ("y_b [m]", np.nan, c['y_b']), ("y_b/(b/2)", np.nan, c['y_b']/L),
                  ("c_b [m]", np.nan, c['c_b']), ("c_t [m]", r0['ct_w'], r['ct_w']),
                  ("xi_mlg", 1.0, e['xi']), ("folga roda-BF [m]", e0['folga'], e['folga']),
                  ("bitola [m]", r0['wheel_span'], r['wheel_span']),
                  ("MAC [m]", e0['MAC'], e['MAC']), ("Xnp AVL [m]", e0['Xnp'], e['Xnp']),
                  ("xcg_aft* [m]", e0['xcg_aft'], e['xcg_aft']),
                  ("SM_aft AVL", e0['SM_aft'], e['SM_aft']), ("SM_fwd AVL", e0['SM_fwd'], e['SM_fwd']),
                  ("tipback* [deg]", e0['tip'], e['tip']),
                  ("folga spray [m]", r0['spray'], r['spray']),
                  ("T0/T0req", r0['T0']/r0['T0req'], r['T0']/r['T0req']),
                  ("landing g", oc.constraint_vector(r0, ['landing'])[0], oc.constraint_vector(r, ['landing'])[0])]:
    print("%-20s %12.4f %12.4f" % (lab, a, b))
print("737: y_b/(b/2)=0.602, ganho raiz +21%%, c_b/c_raiz=0.429 | NJ: %.3f, %+.0f%%, %.3f"
      % (c['y_b']/L, 100*(c['c_root']/r['cr_w'] - 1), c['c_b']/c['c_root']))
json.dump(dict(inputs=inp, y_b=c['y_b'], yf=yf_cfc, c_root=c['c_root']), open(_os.path.join(RESDIR, 'final_design.json'), 'w'), indent=1)
