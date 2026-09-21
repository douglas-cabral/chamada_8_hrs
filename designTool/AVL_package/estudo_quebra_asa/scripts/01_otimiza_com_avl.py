# -*- coding: utf-8 -*-
import json
import os as _os
import sys as _s
_s.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from avl_check import RESDIR
import numpy as np
import opt_avl as oa
import sys as _sys
oa.FUEL_CREDIT = ('--fuel' in _sys.argv)
import opt_nac as _on
_on.CRANK_CB_CT = 2.0 if '--cb2' in _sys.argv else 1.0
TAG = ('fuel' if oa.FUEL_CREDIT else 'cons') + ('_cb2' if '--cb2' in _sys.argv else '')
import opt_common as oc


def clean(d):
    return {k: (clean(v) if isinstance(v, dict) else
                float(v) if isinstance(v, (float, int, np.floating, np.integer)) and not isinstance(v, bool) else v)
            for k, v in d.items()}


W0_REF = oc.run_designTool(oc.get_baseline())['thrust_matching']['W0']
rows = []


def row(tag, inp, r, e, g, cons):
    viol = [(n, v) for n, v in zip(cons, g) if v < -2e-3]
    print("%-8s W0 %7.0f (%+5.2f%%) | S %6.1f AR %5.2f LE %4.1f xr %5.2f | x_n %5.2f y_n %5.2f | "
          "xi %4.1f%% y_b %3.0f%% c_raiz %5.2f | SM_aft(AVL) %5.2f%% | %s"
          % (tag, r['W0']/oc.gravity, 100*(r['W0']/W0_REF - 1), r['S_w'], r['b_w']**2/r['S_w'],
             np.degrees(np.arctan(r['tLE'])), r['xr_w'], r['x_n'], r['y_n'], 100*e['xi'],
             100*e['yb_L'], e['CR'], 100*e['SM_aft'],
             "VIAVEL" if not viol else "INVIAVEL: " + ", ".join("%s %.3f" % v for v in viol)))
    rows.append(dict(tag=tag, inputs=clean(inp), W0=float(r['W0']), SM_aft=float(e['SM_aft']),
                     xi=float(e['xi']), yb_L=float(e['yb_L']), CR=float(e['CR']),
                     feasible=not viol, viol=[(n, float(v)) for n, v in viol],
                     delta=float(oa.STATE['delta'])))
    return not viol


print("A'' : nacele livre + spray, SEM quebra, SM corrigida pelo AVL")
inpA, rA, eA, gA, cA = oa.solve_corrected(oc.get_baseline(), None)
row("A''", inpA, rA, eA, gA, cA)
dA = oa.STATE['delta']

print()
print("Continuacao COM quebra (trem em xi_alvo da corda local), SM corrigida pelo AVL")
inp, d = inpA, dA
for xi in [0.95, 0.90, 0.85, 0.80, 0.775, 0.75, 0.725, 0.70, 0.675, 0.65]:
    print("  alvo xi = %.2f" % xi)
    inp_n, r, e, g, c = oa.solve_corrected(inp, xi, delta0=d)
    ok = row("q%.2f" % xi, inp_n, r, e, g, c)
    if ok:
        inp, d = inp_n, oa.STATE['delta']
    else:
        break

json.dump(rows, open(_os.path.join(RESDIR, 'avl_opt_rows_%s.json' % TAG), 'w'), indent=1)
