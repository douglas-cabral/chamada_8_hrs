# -*- coding: utf-8 -*-
"""AVL trapezoidais (sem Yehudi): atual DT e reopt nacele livre."""
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
TWIST_TIP = -3.0
WING_DZ = -2.140
EH_DZ = 1.600
EV_DZ = 0.250
NAC_Z = -5.0


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


def stations(I, G):
    L = G["b_w"] / 2
    ys = I["D_f"] / 2
    yf = I["b_flap_b_wing"] * L
    ya = (1.0 - I["b_ail_b_wing"]) * L
    ysl = I["b_slat_b_wing"] * L
    raw = [0.0, ys, yf, ya, ysl, L]
    st = []
    for s in raw:
        if not st or abs(s - st[-1]) > 0.55:
            st.append(float(min(max(s, 0.0), L)))
    if abs(st[-1] - L) > 1e-6:
        st.append(float(L))
    return st


def write_avl(path, header, ap, Xref, CDp):
    I, G = ap["inputs"], ap["geometry"]
    L = G["b_w"] / 2
    t = np.tan(ac.change_sweep(0.25, 0.0, I["sweep_w"], L, G["cr_w"], G["ct_w"]))
    dih = I["dihedral_w"]
    zr = I["zr_w"]
    cf = I["c_flap_c_wing"]
    ca = I["c_ail_c_wing"]
    cs = I["c_slat_c_wing"]
    yf = I["b_flap_b_wing"] * L
    ya = (1.0 - I["b_ail_b_wing"]) * L
    ysl = I["b_slat_b_wing"] * L
    ys = I["D_f"] / 2

    def ch(y):
        return G["cr_w"] + (G["ct_w"] - G["cr_w"]) * y / L

    o = []
    w = o.append
    w(header)
    w("# Offsets iguais a fwd/aft: wing Z=-2.14 | EH Z=+1.60 | EV Z=+0.25 | nac Z=-5")
    w("# Planta TRAPEZIO (sem quebra): CR=%.3f ct=%.3f MAC=%.4f" % (
        G["cr_w"], G["ct_w"], G["cm_w"]))
    w("#Mach")
    w(" %.2f" % I["Mach_cruise"])
    w("")
    w("#IYsym IZsym Zsym")
    w(" 0 0 0.0")
    w("")
    w("#Sref Cref Bref")
    w(" %.4f %.4f %.4f" % (I["S_w"], G["cm_w"], G["b_w"]))
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
    for y in stations(I, G):
        c = float(ch(y))
        xle = I["xr_w"] + t * y
        z = zr + y * np.tan(dih)
        twist = TWIST_TIP * y / L
        w("SECTION")
        w("%.4f %.4f %.4f %.4f %.3f" % (xle, y, z, c, twist))
        w("AFILE")
        w("airfoil_lab03.dat")
        if ys - 1e-9 <= y <= ysl + 1e-9:
            w("CONTROL")
            w("slat -1.0 %.3f 0. 0. 0. +1" % (-cs))
        if y <= yf + 1e-9:
            w("CONTROL")
            w("flap 1.0 %.3f 0. 0. 0. +1" % (1.0 - cf))
        if y >= ya - 1e-9:
            w("CONTROL")
            w("aileron -1.0 %.3f 0. 0. 0. -1" % (1.0 - ca))
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


def dump_pair(label, ap, prefix):
    CDp = cruise_cd0(ap)
    xf = ap["balance"]["xcg_fwd"]
    xa = ap["balance"]["xcg_aft"]
    I, G = ap["inputs"], ap["geometry"]
    for tag, xref in (("FWD", xf), ("AFT", xa)):
        name = "%s_%s.avl" % (prefix, tag.lower())
        path = os.path.join(AVLDIR, name)
        header = ("NJ-0502 %s | %s | trapezio sem quebra | airfoil_lab03 | CDp=CD0"
                  % (tag, label))
        write_avl(path, header, ap, xref, CDp)
        print("%s  Xref=%.6f  -> %s" % (name, xref, path))
    print("  S=%.2f  MAC=%.4f  b=%.2f  CR=%.3f  ct=%.3f  x_n=%.3f" % (
        I["S_w"], G["cm_w"], G["b_w"], G["cr_w"], G["ct_w"], I["x_n"]))
    print("  SM_fwd=%.1f%%  SM_aft=%.1f%%  CDp=%.6f" % (
        100 * ap["balance"]["SM_fwd"], 100 * ap["balance"]["SM_aft"], CDp))


def main():
    ap_atual = oc.run_designTool(oc.get_baseline())
    dump_pair("atual DT (my_airplane)", ap_atual, "atual_trap")

    data = json.load(open(os.path.join(RESDIR, "reopt_nacele_livre.json"),
                          encoding="utf-8"))
    ap_reopt = oc.run_designTool(data["inputs"])
    dump_pair("reopt nacele livre", ap_reopt, "reopt_trap")


if __name__ == "__main__":
    main()
