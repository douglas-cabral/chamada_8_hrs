# -*- coding: utf-8 -*-
"""Gera fwd_reopt.avl / aft_reopt.avl no molde dos fwd/aft vigentes."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import avl_check as ac
import opt_common as oc
from designTool.aerodynamics import aerodynamics
from designTool.auxiliary import atmosphere

RESDIR = ac.RESDIR
AVLDIR = ac.AVLDIR
JSON_PATH = os.path.join(RESDIR, "reopt_nacele_livre.json")

YB_FRAC = 0.40
TE_FRAC = 0.10
TWIST_TIP = 0.0
GAP = 0.05
AIL_YA_FRAC = 0.75
WING_DZ = -2.140
EH_DZ = 1.600
EV_DZ = 0.250
NAC_Z = -5.0


def flap_aileron(ap, P):
    I, G = ap["inputs"], ap["geometry"]
    L, ch = P["L"], P["ch"]
    ys = I["D_f"] / 2
    yb = P["Y_B"]
    ya = AIL_YA_FRAC * L
    yf = ya - GAP
    ca_dt = I["c_ail_c_wing"]
    cf_dt = I["c_flap_c_wing"]
    bf = I["b_flap_b_wing"]
    ba = I["b_ail_b_wing"]
    cr0, ct = G["cr_w"], G["ct_w"]

    yf0 = bf * L
    yy = np.linspace(ys, yf0, 4001)
    c_trap = cr0 + (ct - cr0) * yy / L
    Sf_ref = 2.0 * float(np.trapezoid(cf_dt * c_trap, yy))
    yy = np.linspace((1.0 - ba) * L, L, 4001)
    c_trap = cr0 + (ct - cr0) * yy / L
    Sa_ref = 2.0 * float(np.trapezoid(ca_dt * c_trap, yy))

    yy = np.linspace(ya, L, 4001)
    I_c = float(np.trapezoid([ch(v) for v in yy], yy))
    ca = (Sa_ref / 2.0) / I_c

    cb = float(ch(yb))
    yout = np.linspace(yb, yf, 4001)
    I_out = float(np.trapezoid([ch(v) for v in yout], yout))
    c_abs = (Sf_ref / 2.0) / ((yb - ys) + I_out / cb)
    k_cf = c_abs / cb

    def hinge_flap(y):
        if y <= yb + 1e-9:
            return 1.0 - c_abs / ch(y)
        return 1.0 - k_cf

    return dict(ys=ys, yb=yb, yf=yf, ya=ya, YS=I["b_slat_b_wing"] * L,
                c_abs=c_abs, k_cf=k_cf, ca=ca, hinge_flap=hinge_flap,
                hinge_ail=1.0 - ca, hinge_slat=-I["c_slat_c_wing"],
                Sf=Sf_ref, Sa=Sa_ref)


def stations(P, F):
    L = P["L"]
    raw = [0.0, F["ys"], F["yb"], 0.5 * (F["yb"] + F["yf"]),
           F["yf"], F["ya"], F["YS"], L]
    st = []
    for s in raw:
        if s < -1e-9 or s > L + 1e-9:
            continue
        if not st or abs(s - st[-1]) > 0.55:
            st.append(float(s))
    if abs(st[-1] - L) > 1e-6:
        st.append(float(L))
    return st


def cruise_cd0(ap):
    I = ap["inputs"]
    W0 = ap["thrust_matching"]["W0"]
    W_cr = W0 * 0.990 * 0.990 * 0.995 * 0.980
    atm = atmosphere(I["altitude_cruise"])
    V = I["Mach_cruise"] * atm["speed_of_sound"]
    q = 0.5 * atm["density"] * V ** 2
    CL = W_cr / (q * I["S_w"])
    _, _, dd = aerodynamics(
        ap, I["Mach_cruise"], I["altitude_cruise"], CL,
        n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0)
    return float(dd["CD0"])


def write_avl(path, tag, ap, P, F, Xref, CDp, title=None):
    I, G = ap["inputs"], ap["geometry"]
    L, t, ch = P["L"], P["t"], P["ch"]
    dih = I["dihedral_w"]
    zr = I["zr_w"]
    o = []
    w = o.append
    w(title or (
        "NJ-0502 %s | reopt nacele livre | Yehudi yb=40%% te=0.10 | airfoil_lab03 | CDp=CD0" % tag))
    w("# Offsets iguais a fwd/aft: wing Z=-2.14 | EH Z=+1.60 | EV Z=+0.25 | nac Z=-5")
    w("# Planta: CR=%.3f CB=%.3f ct=%.3f MAC_yehudi=%.4f | Cref=MAC_DT=%.4f" % (
        P["CR"], P["CB"], P["ct"], P["MAC"], G["cm_w"]))
    w("#Mach")
    w(" %.2f" % I["Mach_cruise"])
    w("")
    w("#IYsym IZsym Zsym")
    w(" 0 0 0.0")
    w("")
    w("#Sref Cref Bref")
    w(" %.4f %.4f %.4f" % (P["S"], G["cm_w"], 2 * L))
    w("")
    w("#Xref Yref Zref")
    w(" %.6f 0.0 0" % Xref)
    w("")
    w("# CDp")
    w(" %.6f" % CDp)
    w("SURFACE")
    w("Wing")
    w("12 1.0 60 -1.5")
    w("COMPONENT")
    w("1")
    w("YDUPLICATE")
    w("0.0")
    w("TRANSLATE")
    w("0.0 0.0 %.3f" % WING_DZ)
    w("")
    for y in stations(P, F):
        c = float(ch(y))
        xle = P["xr"] + t * y
        z = zr + y * np.tan(dih)
        twist = TWIST_TIP * y / L
        w("SECTION")
        w("%.4f %.4f %.4f %.4f %.3f" % (xle, y, z, c, twist))
        w("AFILE")
        w("airfoil_lab03.dat")
        if F["ys"] - 1e-9 <= y <= F["YS"] + 1e-9:
            w("CONTROL")
            w("slat -1.0 %.3f 0. 0. 0. +1" % F["hinge_slat"])
        if y <= F["yf"] + 1e-9:
            w("CONTROL")
            w("flap 1.0 %.3f 0. 0. 0. +1" % F["hinge_flap"](y))
        if y >= F["ya"] - 1e-9:
            w("CONTROL")
            w("aileron -1.0 %.3f 0. 0. 0. -1" % F["hinge_ail"])
        w("")
    w("SURFACE")
    w("Stab")
    w("8 1.0 16 -1.5")
    w("COMPONENT")
    w("2")
    w("YDUPLICATE")
    w("0.0")
    w("TRANSLATE")
    w("0.0 0.0 %.3f" % EH_DZ)
    w("")
    for (x, y, z, c) in [(G["xr_h"], 0.0, I["zr_h"], G["cr_h"]),
                         (G["xt_h"], G["yt_h"], G["zt_h"], G["ct_h"])]:
        w("SECTION")
        w("%.4f %.4f %.4f %.4f 0.0" % (x, y, z, c))
        w("CONTROL")
        w("elevator 1.0 0.650 0. 0. 0. +1")
        w("")
    w("SURFACE")
    w("Fin")
    w("8 1.0 12 1.0")
    w("COMPONENT")
    w("3")
    w("TRANSLATE")
    w("0.0 0.0 %.3f" % EV_DZ)
    w("")
    for (x, z, c) in [(G["xr_v"], I["zr_v"], G["cr_v"]),
                      (G["xt_v"], G["zt_v"], G["ct_v"])]:
        w("SECTION")
        w("%.4f 0.0 %.4f %.4f 0.0" % (x, z, c))
        w("CONTROL")
        w("rudder 1.0 0.680 0. 0. 0. +1")
        w("")
    w("BODY")
    w("Fuselage")
    w("30 1.0")
    w("SCALE")
    w("%.3f %.3f %.3f" % (I["L_f"], I["D_f"], I["D_f"]))
    w("TRANSLATE")
    w("0.0 0.0 0.000")
    w("BFILE")
    w("fuseB737_nondim.dat")
    w("")
    w("SURFACE")
    w("Nacelle")
    w("6 1.0 12 0.0")
    w("COMPONENT")
    w("4")
    w("YDUPLICATE")
    w("0.0")
    w("SCALE")
    w("%.3f %.3f %.3f" % (I["L_n"], I["D_n"] / 2, I["D_n"] / 2))
    w("TRANSLATE")
    w("%.3f %.3f %.3f" % (I["x_n"], I["y_n"], NAC_Z))
    w("")
    for ang in range(0, 361, 30):
        a = np.radians(ang)
        w("SECTION")
        w(" 0.0 %.4f %.4f 1.0 0. 1 0." % (np.sin(a), np.cos(a)))
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(o) + "\n")


def main():
    data = json.load(open(JSON_PATH, encoding="utf-8"))
    ap = oc.run_designTool(data["inputs"])
    L = ap["geometry"]["b_w"] / 2
    P = ac.planform(ap, YB_FRAC * L, te_frac=TE_FRAC)
    P["Y_B"] = YB_FRAC * L
    F = flap_aileron(ap, P)
    CDp = cruise_cd0(ap)
    xf = ap["balance"]["xcg_fwd"]
    xa = ap["balance"]["xcg_aft"]
    for tag, xref, name in (
        ("FWD", xf, "fwd_reopt.avl"),
        ("AFT", xa, "aft_reopt.avl"),
    ):
        path = os.path.join(AVLDIR, name)
        write_avl(path, tag, ap, P, F, xref, CDp)
        print("%s  Xref=%.6f  -> %s" % (tag, xref, path))
    print("S=%.4f MAC=%.4f b=%.4f CR=%.3f CB=%.3f ct=%.3f" % (
        P["S"], P["MAC"], 2 * P["L"], P["CR"], P["CB"], P["ct"]))
    print("x_n=%.3f y_n=%.3f  CDp=%.6f" % (
        ap["inputs"]["x_n"], ap["inputs"]["y_n"], CDp))
    print("xcg_fwd=%.6f  xcg_aft=%.6f  SM_aft=%.3f" % (
        xf, xa, ap["balance"]["SM_aft"]))


if __name__ == "__main__":
    main()
