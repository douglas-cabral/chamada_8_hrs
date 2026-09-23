# -*- coding: utf-8 -*-
"""Quebra y_b x te_frac (S, BA, c_t fixos) com flap/aileron de area constante.

Lei do flap/aileron (areas = otimizacao DT):
  - flap ys->yb: c_abs constante; yb->yf: cf/c = c_abs/c(yb)
  - aileron: ca/c constante; vao ou razao ajustaveis (AIL_FIT)
  - FLAP_FIT / AIL_FIT: redistribuem corda vs envergadura sem mudar Sf/Sa

Edite YB_FRAC e TE_FRAC abaixo e rode o script.
  Gera: resultados/caso_manual_quebra.json
        resultados/planta_caso_manual.png   (atual vs caso manual)
"""
import copy
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import avl_check as ac
from designTool.analyze import analyze
from designTool.geometry import change_sweep

RESDIR = ac.RESDIR
FINAL = os.path.join(RESDIR, "final_design_dt.json")
GAP = 0.05

# ============ EDITE AQUI ============
YB_FRAC = 0.40    # y_b / (b/2)
TE_FRAC = 0.10    # enflech. BF interno / BA

# Areas SEMPRE = otimizacao DT (Sf_ref, Sa_ref). Abaixo so redistribui corda/vao.
#
# FLAP_FIT:
#   "to_aileron" — yf = ya - GAP; resolve c_abs  (padrao)
#   "span"       — fixe FLAP_YF_FRAC; resolve c_abs
#   "chord"      — fixe FLAP_C_ABS [m]; resolve yf
FLAP_FIT = "to_aileron"
FLAP_YF_FRAC = 0.66   # yf/(b/2) se FLAP_FIT=="span"
FLAP_C_ABS = 2.00     # corda abs. ys->yb [m] se FLAP_FIT=="chord"
#
# AIL_FIT:
#   "ratio" — fixe CA_FRAC (= ca/c); resolve ya  (padrao, CA_FRAC=None usa DT)
#   "span"  — fixe AIL_YA_FRAC; resolve ca/c
AIL_FIT = "span"
CA_FRAC = None        # None -> c_ail_c_wing do DT; senao ca/c manual
AIL_YA_FRAC = 0.75    # ya/(b/2) se AIL_FIT=="span"
# ====================================


def slim(z):
    return {
        k: (float(v) if isinstance(v, (np.floating, float))
            else bool(v) if isinstance(v, (np.bool_, bool)) else v)
        for k, v in z.items()
    }


def draw_case(ax, ap, cfg, title, col, zoom=False):
    I = ap["inputs"]
    P = ac.planform(ap, cfg["yb"], CR=cfg["CR"], te_frac=cfg["te"])
    L = P["L"]
    y = np.linspace(0, L, 1400)
    xle = P["xr"] + P["t"] * y
    c = np.array([P["ch"](v) for v in y])

    ax.plot(np.r_[xle, (xle + c)[::-1]], np.r_[y, y[::-1]],
            "-", color=col, lw=2.0, label="asa")

    yb = cfg["yb"]
    # marca a quebra SO na vista completa (no zoom o axhline/texto
    # saia do eixo e aparecia "solto" entre os paineis)
    if not zoom:
        xle_b = float(P["xr"] + P["t"] * yb)
        cb = float(P["ch"](yb))
        ax.plot([xle_b, xle_b + cb], [yb, yb], "--", color=col, lw=1.4,
                alpha=0.85, zorder=5, clip_on=True)
        ax.annotate(
            r"$y_b$", xy=(xle_b, yb), xytext=(-8, 0),
            textcoords="offset points", color=col, fontsize=10,
            ha="right", va="center", clip_on=True)

    ys = I["D_f"] / 2
    yf = cfg["yf"]
    c_abs, c_end = cfg["c_abs"], cfg["c_end"]
    cb = float(P["ch"](yb))

    def c_flap(yy):
        # ys->yb: absoluto constante; yb->yf: cf/c = c_abs/c(yb)
        if yy <= yb:
            return c_abs
        return c_abs * float(P["ch"](yy)) / cb

    yf_line = np.linspace(ys, yf, 500)
    xle_f = P["xr"] + P["t"] * yf_line
    cf = np.array([c_flap(v) for v in yf_line])
    cw = np.array([P["ch"](v) for v in yf_line])
    ax.plot(xle_f + cw - cf, yf_line, ":", color=col, lw=1.8,
            label="dobradica flap", clip_on=True)

    ya = cfg["ya"]
    ca = float(cfg.get("ca", I["c_ail_c_wing"]))
    ya_line = np.linspace(ya, L, 200)
    xle_a = P["xr"] + P["t"] * ya_line
    ca_w = np.array([P["ch"](v) for v in ya_line])
    ax.plot(xle_a + (1 - ca) * ca_w, ya_line, "-.", color=col, lw=1.2,
            alpha=0.85, label="dobradica aileron", clip_on=True)

    ax.add_patch(plt.Rectangle(
        (I["x_n"], I["y_n"] - I["D_n"] / 2), I["L_n"], I["D_n"],
        fc="none", ec="0.35", lw=1.2, ls="-", zorder=4, clip_on=True))
    ax.add_patch(plt.Rectangle(
        (0, -0.2), I["L_f"], I["D_f"] / 2 + 0.2,
        fc="0.93", ec="0.6", zorder=0, alpha=0.9, clip_on=True))

    y_m = I["y_mlg"]
    c_m = P["ch"](y_m)
    xle_m = P["xr"] + P["t"] * y_m
    xi_h = 1.0 - c_abs / c_m
    xi_pin = xi_h - 0.02
    x_pin = xle_m + xi_pin * c_m
    x_w = I["x_mlg"]
    x_h = xle_m + xi_h * c_m
    trail = x_w - x_pin

    dl = 0.22 * I["D_f"]
    ax.add_patch(plt.Rectangle(
        (x_w - dl / 2, y_m - 0.32), dl, 0.64,
        fc=col, ec="k", alpha=0.85, zorder=6, clip_on=True))
    ax.plot(x_pin, y_m, "x", color=col, ms=11, mew=2.4, zorder=7, clip_on=True)
    ax.plot(x_h, y_m, "o", mfc="none", mec=col, ms=8, mew=1.4, zorder=7,
            clip_on=True)
    ax.plot([x_pin, x_w], [y_m, y_m], color=col, lw=2.0, zorder=6, clip_on=True)
    ax.plot([xle_m, xle_m + c_m], [y_m, y_m], color="0.25", lw=0.9,
            alpha=0.6, zorder=3, clip_on=True)

    if zoom:
        ax.set_xlim(xle_m - 1.5, xle_m + c_m + 1.0)
        ax.set_ylim(y_m - 3.5, y_m + 4.5)
    else:
        x_min = float(np.min(xle)) - 1.0
        x_max = float(np.max(xle + c)) + 1.0
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, L + 0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_clip_on(True)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    info = (
        r"$y_b$=%.1f%%  te_frac=%.2f" "\n"
        r"$\varepsilon_{roda}$=%.3f  trail=%.2f m" "\n"
        r"$c_{flap}$=%.2f m  $c_R$=%.2f  $c_b$=%.2f" "\n"
        r"$S_f$=%.1f  $S_a$=%.1f m$^2$  pack=%s"
        % (100 * cfg["ybf"], cfg["te"], cfg["xi_w"], trail,
           c_abs, cfg["CR"], cfg["CB"], cfg["Sf"], cfg["Sa"],
           "OK" if cfg.get("pack_ok") else "nao")
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace", clip_on=True,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))


def main():
    fin = json.load(open(FINAL, encoding="utf-8"))
    I = copy.deepcopy(fin["inputs"])
    ap = {"inputs": I}
    analyze(ap, False, False)
    G = ap["geometry"]
    L = float(G["b_w"] / 2)
    ct = float(G["ct_w"])
    S2 = float(I["S_w"] / 2)
    cr0 = float(G["cr_w"])
    t = float(np.tan(change_sweep(0.25, 0.0, I["sweep_w"], L, cr0, ct)))
    ys = float(I["D_f"] / 2)
    y_m = float(I["y_mlg"])
    x_mlg = float(I["x_mlg"])
    xr = float(I["xr_w"])
    ca = float(I["c_ail_c_wing"])
    cf = float(I["c_flap_c_wing"])
    bf = float(I["b_flap_b_wing"])
    ba = float(I["b_ail_b_wing"])

    # Areas de referencia = otimizacao DT na asa TRAPEZOIDAL (mesmos cf, bf, ca, ba)
    yf_opt = bf * L
    ya_opt = (1.0 - ba) * L
    yy = np.linspace(ys, yf_opt, 5001)
    c_trap = cr0 + (ct - cr0) * yy / L
    Sf_ref = 2.0 * float(np.trapezoid(cf * c_trap, yy))
    yy = np.linspace(ya_opt, L, 5001)
    c_trap = cr0 + (ct - cr0) * yy / L
    Sa_ref = 2.0 * float(np.trapezoid(ca * c_trap, yy))

    Sf_target = float(Sf_ref)  # area de flap sempre a da otimizacao
    if FLAP_FIT not in ("to_aileron", "span", "chord"):
        raise ValueError('FLAP_FIT deve ser "to_aileron", "span" ou "chord"')
    if AIL_FIT not in ("ratio", "span"):
        raise ValueError('AIL_FIT deve ser "ratio" ou "span"')
    ca_default = float(CA_FRAC) if CA_FRAC is not None else ca

    print("Refs OTIMIZACAO DT (trapezio): Sf=%.3f m2  Sa=%.3f m2"
          % (Sf_ref, Sa_ref))
    print("  (cf/c=%.2f bf/b=%.2f | ca/c=%.2f ba/b=%.2f)"
          % (cf, bf, ca, ba))
    print("Redistribuicao: FLAP_FIT=%s  AIL_FIT=%s" % (FLAP_FIT, AIL_FIT))

    def geom(yb, te_frac):
        s = te_frac * t - t
        den = L + yb
        CR = (2 * S2 - L * ct + yb * ct - s * yb * L) / den
        CB = CR + s * yb
        if not (np.isfinite(CR) and CR > 0.5 and CB >= ct - 1e-6
                and (y_m + 1.0) < yb < 0.98 * L):
            return None

        def ch(y):
            if y <= yb:
                return CR + s * y
            return CB + (ct - CB) * (y - yb) / max(L - yb, 1e-9)

        return dict(CR=float(CR), CB=float(CB), ch=ch)

    def aileron_fit(ch):
        """Area Sa_ref. ratio: fixa ca, resolve ya | span: fixa ya, resolve ca."""
        if AIL_FIT == "ratio":
            ca_use = ca_default
            lo, hi = ys + 2.0, L - 0.3
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                yy = np.linspace(mid, L, 4001)
                Sa = 2 * float(np.trapezoid([ca_use * ch(v) for v in yy], yy))
                if Sa > Sa_ref:
                    lo = mid
                else:
                    hi = mid
            ya = 0.5 * (lo + hi)
            yy = np.linspace(ya, L, 4001)
            Sa = 2 * float(np.trapezoid([ca_use * ch(v) for v in yy], yy))
            return float(ya), float(ca_use), float(Sa)

        ya = float(AIL_YA_FRAC) * L
        if not (ys + 1.0 < ya < L - 0.2):
            return None
        yy = np.linspace(ya, L, 4001)
        I_c = float(np.trapezoid([ch(v) for v in yy], yy))
        if I_c <= 1e-12:
            return None
        ca_use = (Sa_ref / 2.0) / I_c
        if not (0.15 < ca_use < 0.55):
            return None
        Sa = 2.0 * ca_use * I_c
        return float(ya), float(ca_use), float(Sa)

    def flap_area(ch, yb, yf, c_abs):
        cb = float(ch(yb))
        k = c_abs / cb
        yin = np.linspace(ys, yb, 2001)
        yout = np.linspace(yb, yf, 4001)
        Ain = float(np.trapezoid(np.full_like(yin, c_abs), yin))
        Aout = float(np.trapezoid([k * ch(v) for v in yout], yout))
        return 2.0 * (Ain + Aout), k

    def flap_fit(ch, yb, yf_max, sf_tgt):
        """Lei: ys->yb c_abs const; yb->yf cf/c=c_abs/c(yb). Area = sf_tgt."""
        cb = float(ch(yb))
        if cb <= 1e-9 or yf_max <= yb + 0.8:
            return None
        c_in_min = min(float(ch(y)) for y in np.linspace(ys, yb, 801))

        if FLAP_FIT == "to_aileron":
            yf = float(yf_max)
            yout = np.linspace(yb, yf, 4001)
            I_out = float(np.trapezoid([ch(v) for v in yout], yout))
            den = (yb - ys) + I_out / cb
            if den <= 1e-12:
                return None
            c_abs = (sf_tgt / 2.0) / den

        elif FLAP_FIT == "span":
            yf = float(FLAP_YF_FRAC) * L
            if yf > yf_max + 1e-9 or yf < yb + 0.8:
                return None
            yout = np.linspace(yb, yf, 4001)
            I_out = float(np.trapezoid([ch(v) for v in yout], yout))
            den = (yb - ys) + I_out / cb
            if den <= 1e-12:
                return None
            c_abs = (sf_tgt / 2.0) / den

        else:  # chord
            c_abs = float(FLAP_C_ABS)
            if not (0.35 < c_abs <= min(c_in_min, cb) + 1e-6):
                return None
            lo, hi = yb + 0.8, float(yf_max)
            Sf_lo, _ = flap_area(ch, yb, lo, c_abs)
            Sf_hi, _ = flap_area(ch, yb, hi, c_abs)
            if Sf_tgt < Sf_lo - 0.15 or Sf_tgt > Sf_hi + 0.15:
                return None
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                Sf_m, _ = flap_area(ch, yb, mid, c_abs)
                if Sf_m < sf_tgt:
                    lo = mid
                else:
                    hi = mid
            yf = 0.5 * (lo + hi)

        if not (0.35 < c_abs <= min(c_in_min, cb) + 1e-6):
            return None
        Sf, k = flap_area(ch, yb, yf, c_abs)
        if abs(Sf - sf_tgt) > 0.15:
            return None
        c_end = k * float(ch(yf))
        xi_h = 1.0 - c_abs / float(ch(y_m))
        return dict(c_abs=float(c_abs), c_end=float(c_end), k_cf=float(k),
                    xi_h=float(xi_h), Sf=float(Sf), yf=float(yf))

    def evaluate(yb, te):
        g = geom(yb, te)
        if g is None:
            return dict(ok=False, reason="geom")
        ch = g["ch"]
        ail = aileron_fit(ch)
        if ail is None:
            return dict(ok=False, reason="aileron")
        ya, ca_use, Sa = ail
        yf_max = ya - GAP
        if yf_max < yb + 0.8:
            return dict(ok=False, reason="yf_yb")
        ff = flap_fit(ch, yb, yf_max, Sf_target)
        if ff is None:
            return dict(ok=False, reason="flap")
        yf = ff["yf"]
        c_m = float(ch(y_m))
        xle = xr + t * y_m
        xi_w = (x_mlg - xle) / c_m
        xi_pin = ff["xi_h"] - 0.02
        trail = (xi_w - xi_pin) * c_m
        pack = ((trail > 0.0)
                and (0.55 <= xi_pin <= ff["xi_h"] - 0.005)
                and (xi_w < 0.98))
        return dict(
            ok=True, pack_ok=pack, yb=float(yb), ybf=float(yb / L), te=float(te),
            CR=g["CR"], CB=g["CB"], c_m=c_m, xi_w=float(xi_w),
            xi_h=ff["xi_h"], xi_pin=float(xi_pin), trail=float(trail),
            c_abs=ff["c_abs"], c_end=ff["c_end"], k_cf=ff["k_cf"],
            ya=float(ya), yf=float(yf), ca=float(ca_use),
            Sa=Sa, Sf=ff["Sf"], Sf_target=Sf_target,
            FLAP_FIT=FLAP_FIT, AIL_FIT=AIL_FIT)

    cur = evaluate(float(fin["y_b"]), float(fin["te_frac"]))

    yb = float(YB_FRAC) * L
    te = float(TE_FRAC)
    print("=" * 64)
    print(" MANUAL: YB_FRAC=%.3f  TE_FRAC=%.3f" % (YB_FRAC, TE_FRAC))
    print(" FLAP_FIT=%s  AIL_FIT=%s | Sf=Sa refs DT (%.2f / %.2f m2)"
          % (FLAP_FIT, AIL_FIT, Sf_ref, Sa_ref))
    print(" Lei flap: c_abs ys->yb; cf/c const yb->yf | areas fixas")
    print("=" * 64)
    if yb < y_m + 1.0:
        print("ERRO: y_b=%.2f m < y_mlg+1=%.2f m" % (yb, y_m + 1.0))
        return
    man = evaluate(yb, te)
    if not man.get("ok"):
        print("ERRO: caso inviavel (%s). Ajuste FLAP_FIT/AIL_FIT/YB/TE."
              % man.get("reason", "?"))
        return

    print("ATUAL : yb=%.1f%% te=%.2f  Sf=%.1f Sa=%.1f  trail=%.2f"
          % (100 * cur["ybf"], cur["te"], cur["Sf"], cur["Sa"], cur["trail"]))
    print("MANUAL: yb=%.1f%% te=%.2f  Sf=%.1f Sa=%.1f  trail=%.2f  pack=%s"
          % (100 * man["ybf"], man["te"], man["Sf"], man["Sa"], man["trail"],
             man["pack_ok"]))
    print("        c_abs=%.2f  cf/c_out=%.3f  yf%%=%.1f  ca/c=%.3f  ya%%=%.1f"
          % (man["c_abs"], man["k_cf"], 100 * man["yf"] / L,
             man["ca"], 100 * man["ya"] / L))
    print("        xi_pin=%.3f  CR=%.2f  CB=%.2f"
          % (man["xi_pin"], man["CR"], man["CB"]))

    out_json = os.path.join(RESDIR, "caso_manual_quebra.json")
    json.dump(dict(
        refs=dict(Sf=Sf_ref, Sa=Sa_ref, GAP=GAP),
        current=slim(cur),
        manual=slim(man),
        params=dict(
            YB_FRAC=YB_FRAC, TE_FRAC=TE_FRAC,
            FLAP_FIT=FLAP_FIT, FLAP_YF_FRAC=FLAP_YF_FRAC,
            FLAP_C_ABS=FLAP_C_ABS, AIL_FIT=AIL_FIT,
            CA_FRAC=CA_FRAC, AIL_YA_FRAC=AIL_YA_FRAC),
        flap_law="ys->yb: c_abs const; yb->yf: cf/c = c_abs/c(yb)",
        area_policy="Sf/Sa = areas da otimizacao DT (trapezio)",
    ), open(out_json, "w", encoding="utf-8"), indent=2)

    fig = plt.figure(figsize=(16.0, 14.5))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.4, 1.0], hspace=0.28, wspace=0.18,
        left=0.05, right=0.98, top=0.93, bottom=0.05)
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    draw_case(axes[0, 0], ap, cur, "Atual (referencia)", "C0", zoom=False)
    draw_case(axes[0, 1], ap, man,
              "Manual  yb=%.0f%%  te=%.2f" % (100 * YB_FRAC, TE_FRAC),
              "C3", zoom=False)
    draw_case(axes[1, 0], ap, cur, "Zoom MLG — Atual", "C0", zoom=True)
    draw_case(axes[1, 1], ap, man, "Zoom MLG — Manual", "C3", zoom=True)
    axes[1, 0].plot([], [], "x", color="k", ms=8, mew=2, label="pino")
    axes[1, 0].plot([], [], "o", mfc="none", mec="k", ms=7, label="dobradica flap")
    axes[1, 0].plot([], [], "s", color="k", ms=7, label="roda")
    axes[1, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Caso manual vs atual  |  $S_w$, BA, $c_t$ fixos  |  "
        "flap/aileron com area mantida",
        fontsize=13)
    out_png = os.path.join(RESDIR, "planta_caso_manual.png")
    fig.savefig(out_png, dpi=170)
    print(out_png)
    print(out_json)


if __name__ == "__main__":
    main()
