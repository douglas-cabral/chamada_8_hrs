# -*- coding: utf-8 -*-
import json
import os as _os
import sys as _s
_s.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from avl_check import RESDIR
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import opt_avl as oa
import opt_common as oc
import avl_check as ac

d = json.load(open(_os.path.join(RESDIR, 'final_design.json')))
cases = [("Lab 02 (trapezoidal)", oc.get_baseline(), None, 'C0', '--'),
         ("re-otimizada + quebra", d['inputs'], d['y_b'], 'C3', '-')]
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1.35, 1]})
for ax in axes:
    for nm, inp, yb, col, ls in cases:
        ap = oc.run_designTool(inp); I, G = ap['inputs'], ap['geometry']
        L = G['b_w']/2
        P = ac.planform(ap, yb if yb else 1e-6)
        y = np.linspace(0, L, 1500)
        xle = P['xr'] + P['t']*y
        c = np.array([P['ch'](v) for v in y])
        ax.plot(np.r_[xle, (xle + c)[::-1]], np.r_[y, y[::-1]], ls, color=col, lw=2, label=nm)
        yf = I['b_flap_b_wing']*L
        m = (y >= I['D_f']/2) & (y <= yf)
        ax.plot((xle + 0.68*c)[m], y[m], ':', color=col, lw=1.3)
        dl = 0.2*I['D_f']
        ax.add_patch(plt.Rectangle((I['x_mlg'] - dl/2, I['y_mlg'] - 0.35), dl, 0.7,
                                   fc=col, ec='k', alpha=0.8, zorder=6))
        ax.add_patch(plt.Rectangle((I['x_n'], I['y_n'] - I['D_n']/2), I['L_n'], I['D_n'],
                                   fc='none', ec=col, lw=1.4, ls=ls, zorder=5))
        yy = np.array([0, 20])
        ax.plot(I['x_nlg'] + yy/np.tan(np.radians(22)), yy, color=col, lw=0.8, alpha=0.6)
    ax.add_patch(plt.Rectangle((0, -3), 65.09, 3 + 5.96/2, fc='0.93', ec='0.6', zorder=0))
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
axes[0].set_xlim(0, 52); axes[0].set_ylim(-1, 35)
axes[0].set_title('Vista em planta (semi-asa)\nretangulos cheios = trem | vazados = nacele | linhas finas = cone de spray 22°')
axes[0].legend(loc='upper left')
axes[1].set_xlim(16, 40); axes[1].set_ylim(-1, 18)
axes[1].set_title('Zoom: trem principal\nLab 02: xi = 100% (roda passa do BF)   |   novo: xi = 70%')
fig.tight_layout()
out = os.path.join(ac.AVLDIR, 'nj0502_otim_quebra_planta.png')
fig.savefig(out, dpi=150); print(out)
