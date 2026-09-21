# -*- coding: utf-8 -*-
"""Gera .avl completo (asa com quebra + EH + EV + fuselagem + nacele) a partir de
um dicionario de inputs e roda o AVL para obter Xnp e a margem estatica."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AVLDIR = os.path.abspath(os.path.join(HERE, '..', '..'))            # designTool/AVL_package
DESIGNTOOL_DIR = os.path.abspath(os.path.join(AVLDIR, '..'))        # designTool/
RESDIR = os.path.abspath(os.path.join(HERE, '..', 'resultados'))
sys.path.insert(0, DESIGNTOOL_DIR)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opt_nac as on  # noqa: F401  (registra o path do Lab 02 e as restricoes novas)
import opt_common as oc
from designTool.geometry import change_sweep

TWIST_TIP = -3.0


def planform(ap, Y_B):
    I, G = ap['inputs'], ap['geometry']
    L = G['b_w']/2; cr0 = G['cr_w']; ct = G['ct_w']; xr = I['xr_w']
    t = np.tan(change_sweep(0.25, 0., I['sweep_w'], L, cr0, ct))
    A = Y_B + (L - Y_B)/2.
    B = -t*Y_B**2/2. + (L - Y_B)*(-t*Y_B + ct)/2.
    CR = (I['S_w']/2 - B)/A
    CB = CR - t*Y_B

    def ch(y):
        return CR - t*y if y <= Y_B else CB + (ct - CB)*(y - Y_B)/(L - Y_B)
    y = np.linspace(0, L, 100001)
    c = np.array([ch(v) for v in y])
    a = np.trapezoid(c, y)
    mac = np.trapezoid(c*c, y)/a
    xm = np.trapezoid(c*(xr + t*y), y)/a
    m = y <= 0.98*L
    tc = I['tcr_w'] + (I['tct_w'] - I['tcr_w'])*y/L
    wg = (c**2*tc)[m]
    xf = (xr + t*y + (I['x_tank_c_w'] + 0.5*I['c_tank_c_w'])*c)[m]
    vol = np.trapezoid(wg, y[m])
    return dict(L=L, t=t, CR=CR, CB=CB, ct=ct, xr=xr, ch=ch, S=2*a, MAC=mac,
                XM=xm, xcg_w=xm + 0.4*mac,
                xcg_f=np.trapezoid(wg*xf, y[m])/vol, vol=vol)


def write_avl(ap, P, Xref, path, yf=None, header=None):
    I, G = ap['inputs'], ap['geometry']
    L, t, ch = P['L'], P['t'], P['ch']
    dih = I['dihedral_w']; zr = I['zr_w']
    cf = I['c_flap_c_wing']
    # flap: area constante (convencao designTool) com cf/c constante
    cr0, ct = G['cr_w'], G['ct_w']
    yf0 = I['b_flap_b_wing']*L
    Sf0 = cf*yf0*(cr0 + (cr0 - (cr0 - ct)*I['b_flap_b_wing']))/2.
    lo, hi = 0.1, L
    for _ in range(100):
        mid = (lo + hi)/2
        gg = np.linspace(0, mid, 4001)
        if cf*np.trapezoid([ch(v) for v in gg], gg) < Sf0:
            lo = mid
        else:
            hi = mid
    YF = (lo + hi)/2
    if yf is not None:
        YF = yf
    YA = (1 - I['b_ail_b_wing'])*L
    YS = I['b_slat_b_wing']*L
    YSOB = I['D_f']/2
    Y_B = P['Y_B']
    raw = sorted([0.0, YSOB, Y_B, YF, YA, YS, L])
    st = []
    for s in raw:
        if not st or s - st[-1] > 0.6:
            st.append(s)
    o = []
    w = o.append
    w("NJ-0502 - asa com quebra (y_b=%.3f m)" % Y_B)
    for hl in (header or []):
        w("#  " + hl)
    w("#Mach"); w(" %.2f" % I['Mach_cruise']); w("")
    w("#IYsym IZsym Zsym"); w(" 0 0 0.0"); w("")
    w("#Sref Cref Bref"); w(" %.4f %.4f %.4f" % (P['S'], P['MAC'], 2*L)); w("")
    w("#Xref Yref Zref"); w(" %.4f 0.0 %.4f" % (Xref, zr)); w("")
    w("# CDp"); w(" 0.0"); w("")
    w("SURFACE"); w("Wing"); w("12 1.0 60 -1.5"); w("COMPONENT"); w("1")
    w("YDUPLICATE"); w("0.0"); w("")
    for y in st:
        w("SECTION")
        w("%.4f %.4f %.4f %.4f %.3f" % (P['xr'] + t*y, y, zr + y*np.tan(dih), ch(y), TWIST_TIP*y/L))
        w("AFILE"); w("a1.dat")
        if YSOB - 1e-9 <= y <= YS + 1e-9:
            w("CONTROL"); w("slat -1.0 %.3f 0. 0. 0. +1" % (-I['c_slat_c_wing']))
        if y <= YF + 1e-9:
            w("CONTROL"); w("flap 1.0 %.3f 0. 0. 0. +1" % (1 - cf))
        if y >= YA - 1e-9:
            w("CONTROL"); w("aileron -1.0 %.3f 0. 0. 0. -1" % (1 - I['c_ail_c_wing']))
        w("")
    w("SURFACE"); w("Stab"); w("8 1.0 16 -1.5"); w("COMPONENT"); w("2")
    w("YDUPLICATE"); w("0.0"); w("")
    for (x, y, z, c) in [(G['xr_h'], 0.0, I['zr_h'], G['cr_h']),
                         (G['xt_h'], G['yt_h'], G['zt_h'], G['ct_h'])]:
        w("SECTION"); w("%.4f %.4f %.4f %.4f 0.0" % (x, y, z, c))
        w("CONTROL"); w("elevator 1.0 0.650 0. 0. 0. +1"); w("")
    w("SURFACE"); w("Fin"); w("8 1.0 12 1.0"); w("COMPONENT"); w("3"); w("")
    for (x, z, c) in [(G['xr_v'], I['zr_v'], G['cr_v']), (G['xt_v'], G['zt_v'], G['ct_v'])]:
        w("SECTION"); w("%.4f 0.0 %.4f %.4f 0.0" % (x, z, c))
        w("CONTROL"); w("rudder 1.0 0.680 0. 0. 0. +1"); w("")
    w("BODY"); w("Fuselage"); w("30 1.0"); w("SCALE")
    w("%.3f %.3f %.3f" % (I['L_f'], I['D_f'], I['D_f'])); w("BFILE"); w("fuseB737_nondim.dat"); w("")
    w("SURFACE"); w("Nacelle"); w("6 1.0 12 0.0"); w("COMPONENT"); w("4")
    w("YDUPLICATE"); w("0.0"); w("SCALE")
    w("%.3f %.3f %.3f" % (I['L_n'], I['D_n']/2, I['D_n']/2))
    w("TRANSLATE"); w("%.3f %.3f %.3f" % (I['x_n'], I['y_n'], I['z_n'])); w("")
    for ang in range(0, 361, 30):
        a = np.radians(ang)
        w("SECTION"); w(" 0.0 %.4f %.4f 1.0 0. 1 0." % (np.sin(a), np.cos(a)))
    open(path, 'w').write("\n".join(o) + "\n")
    return YF


def xnp(name):
    p = subprocess.run([os.path.join(AVLDIR, "avl337.exe")],
                       input="load %s\noper\na\na\n2.0\nx\nst\n\n" % name,
                       capture_output=True, text=True, cwd=AVLDIR, timeout=600)
    for ln in p.stdout.splitlines():
        if "Xnp" in ln:
            return float(ln.split("=")[1])
    raise RuntimeError(p.stdout[-1500:])


def evaluate(inputs, Y_B, name, keep=False, fuel_credit=None):
    if fuel_credit is None:
        import opt_avl
        fuel_credit = opt_avl.FUEL_CREDIT
    ap = oc.run_designTool(inputs)
    P0 = planform(ap, 1e-6)
    P = planform(ap, Y_B); P['Y_B'] = Y_B
    W0 = ap['thrust_matching']['W0']
    dcg = (ap['empty_weight']['W_w']*(P['xcg_w'] - P0['xcg_w'])
           + (ap['thrust_matching']['W_fuel']*(P['xcg_f'] - P0['xcg_f']) if fuel_credit else 0.0))/W0
    xa = ap['balance']['xcg_aft'] + dcg
    xf = ap['balance']['xcg_fwd'] + dcg
    path = os.path.join(AVLDIR, name)
    YF = write_avl(ap, P, xa, path)
    X = xnp(name)
    if not keep:
        os.remove(path)
    I = ap['inputs']
    xle = P['xr'] + P['t']*I['y_mlg']
    c_m = P['ch'](I['y_mlg'])
    return dict(W0=W0, Xnp=X, xcg_aft=xa, xcg_fwd=xf, dcg=dcg, MAC=P['MAC'],
                SM_aft=(X - xa)/P['MAC'], SM_fwd=(X - xf)/P['MAC'],
                xi=(I['x_mlg'] - xle)/c_m, CR=P['CR'], CB=P['CB'], ct=P['ct'],
                yb_L=Y_B/P['L'], YF_L=YF/P['L'],
                folga=xle + c_m - (I['x_mlg'] + 0.1*I['D_f']),
                dvol=P['vol']/P0['vol'] - 1,
                tip=np.degrees(np.arctan((I['x_mlg'] - xa)/(-I['z_lg']))),
                nlg_aft=(I['x_mlg'] - xa)/(I['x_mlg'] - I['x_nlg']),
                nlg_fwd=(I['x_mlg'] - xf)/(I['x_mlg'] - I['x_nlg']))
