# -*- coding: utf-8 -*-
"""
Re-otimizacao da NJ-0502 com nacele livre (x_n, y_n) sobre o Lab 02.

Estende DESIGN_VARS / CONSTRAINTS de opt_common em tempo de execucao
(so nacele: spray, nac_*). O DT permanece trapezoidal.

A quebra da asa nao entra aqui: e ajustada a posteriori em quebra_asa.py.
"""
import copy
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
AVLDIR = os.path.abspath(os.path.join(HERE, '..', '..'))            # designTool/AVL_package
DESIGNTOOL_DIR = os.path.abspath(os.path.join(AVLDIR, '..'))        # designTool/
RESDIR = os.path.abspath(os.path.join(HERE, '..', 'resultados'))

OPTDIR = os.path.join(DESIGNTOOL_DIR, 'lab02_opt', 'otimizacao_NJ0502')
sys.path.insert(0, OPTDIR)

import opt_common as oc

deg2rad = oc.deg2rad
rad2deg = oc.rad2deg

SPRAY_DEG = 22.0          # semi-angulo do cone de spray do trem de nariz
DZ_NAC = 2.726            # nacele abaixo do BA da asa (mantido da linha de base)

# ---------------------------------------------------------------- variaveis
oc.DESIGN_VARS.append(('x_n', r'$x_n$', 'm', 1.0, 14.0, 27.0))
oc.DESIGN_VARS.append(('y_n', r'$y_n$', 'm', 1.0,  5.5, 14.0))
oc.DV_INDEX = {s[0]: i for i, s in enumerate(oc.DESIGN_VARS)}
oc.DV_NAMES = [s[0] for s in oc.DESIGN_VARS]

_extract0 = oc.extract


def extract(airplane):
    r = _extract0(airplane)
    inp, geo = airplane['inputs'], airplane['geometry']
    L = geo['b_w']/2.0
    eta_n = min(inp['y_n']/L, 1.0)
    x_le_n = inp['xr_w'] + eta_n*(geo['xt_w'] - inp['xr_w'])
    r['x_n'] = inp['x_n']
    r['y_n'] = inp['y_n']
    r['z_n'] = inp['z_n']
    r['L_n'] = inp['L_n']
    r['D_n'] = inp['D_n']
    r['x_nlg'] = inp['x_nlg']
    r['z_lg'] = inp['z_lg']
    r['y_mlg'] = inp['y_mlg']
    r['x_le_n'] = x_le_n
    # margem do cone de spray: positivo = entrada da nacele FORA do cone.
    # Ponto critico = canto interno da boca (x_n, y_n - D_n/2): e o y menor
    # da entrada, onde a linha de 22° esta mais a frente.
    y_in = inp['y_n'] - 0.5*inp['D_n']
    r['spray'] = (inp['x_nlg'] + y_in/np.tan(SPRAY_DEG*deg2rad) - inp['x_n'])
    return r


oc.extract = extract

# ---------------------------------------------------------------- restricoes (so nacele)
oc.CONSTRAINTS.append((
    'spray', r'entrada interna da nacele fora do cone de spray de $22^\circ$',
    r'$(x_{nlg}+(y_n-D_n/2)/\tan 22^\circ-x_n)/L_f$',
    lambda r: r['spray']/r['L_f'], 'adicionada'))
# Saida do eixo da nacele nao pode ficar a frente do BA em y_n
# (pode ir para tras / sob a asa). g = (x_n+L_n - x_LE)/L_n >= 0
oc.CONSTRAINTS.append((
    'nac_aft_le',
    r'saida do eixo da nacele $\geq$ BA em $y_n$',
    r'$(x_n+L_n-x_{LE,n})/L_n$',
    lambda r: (r['x_n'] + r['L_n'] - r['x_le_n'])/r['L_n'], 'adicionada'))
oc.CONSTRAINTS.append((
    'nac_gnd', r'nacele $\geq 0{,}5$ m do solo',
    r'$(z_n-D_n/2-z_{lg}-0{,}5)/D_n$',
    lambda r: (r['z_n'] - r['D_n']/2 - r['z_lg'] - 0.5)/r['D_n'], 'adicionada'))

# motor por fora do trem principal: borda interna da nacele >= y_mlg + 1,0 m
oc.CONSTRAINTS.append((
    'nac_mlg', r'nacele por fora do trem ($y_n-D_n/2 \geq y_{mlg}+1$ m)',
    r'$(y_n-D_n/2-y_{mlg}-1)/D_n$',
    lambda r: (r['y_n'] - r['D_n']/2 - r['y_mlg'] - 1.0)/r['D_n'], 'adicionada'))

oc.CON_NAMES = [c[0] for c in oc.CONSTRAINTS]
oc.CON_INDEX = {c[0]: i for i, c in enumerate(oc.CONSTRAINTS)}


class Model(oc.Model):
    """Model com z_n acompanhando a asa (mantem o offset vertical da nacele)."""

    def build_inputs(self, x):
        inputs = copy.deepcopy(self.base_inputs)
        for name, value in zip(self.dv_names, self.to_physical(x)):
            inputs[name] = value
        dih = inputs['dihedral_w']
        inputs['z_n'] = inputs['zr_w'] + inputs['y_n']*np.tan(dih) - DZ_NAC
        return inputs


def solve(tag, xi_max=1.00, extra_cons=None, dv=None, cons=None):
    """SLSQP trapezoidal com nacele livre. Sem restricoes de quebra."""
    oc.XI_MLG_MAX = xi_max
    names = dv if dv is not None else oc.DV_NAMES
    con = list(cons) if cons is not None else list(oc.CON_NAMES) + list(extra_cons or [])
    model = Model(names, con_names=con)
    res = minimize(model.objfun, model.x0, jac=model.objgrad,
                   constraints=[{'type': 'ineq', 'fun': model.confun,
                                 'jac': model.conjac}],
                   bounds=model.bounds, method='slsqp',
                   options={'maxiter': 300, 'ftol': 1e-6, 'disp': False})
    r = model.results(res.x)
    oc.XI_MLG_MAX = 1.00
    return model, res, r


def report(tag, model, res, r, dv_names):
    print("=" * 96)
    print(" %s" % tag)
    print("=" * 96)
    print("  %s | n_f=%d | W0 = %.0f kgf  (%+.2f%% vs baseline)"
          % (res.message, model.n_objfun, r['W0']/oc.gravity,
             100.0*(r['W0']/model.W0_ref - 1.0)))
    print("  VARIAVEIS:")
    for name, v0, v in zip(dv_names, model.to_physical(model.x0),
                           model.to_physical(res.x)):
        spec = [s for s in oc.DESIGN_VARS if s[0] == name][0]
        f = spec[3]
        lo, hi = spec[4]*f, spec[5]*f
        at = ""
        if abs(v*f - lo) < 1e-3*max(1, abs(lo)):
            at = " <-- no limite inferior"
        if abs(v*f - hi) < 1e-3*max(1, abs(hi)):
            at = " <-- no limite superior"
        print("    %-9s %10.4f -> %10.4f %-6s [%.2f, %.2f]%s"
              % (name, v0*f, v*f, spec[2], lo, hi, at))
    g = oc.constraint_vector(r, model.con_names)
    ativas = [n for n, gv in zip(model.con_names, g) if abs(gv) < 1e-4]
    viol = [(n, gv) for n, gv in zip(model.con_names, g) if gv < -1e-4]
    print("  ATIVAS (%d): %s" % (len(ativas), ", ".join(ativas)))
    if viol:
        print("  VIOLADAS: %s" % ", ".join("%s=%.4f" % v for v in viol))
    print("  S_w=%.2f m2 | AR=%.3f | b=%.2f m | xi_mlg=%.4f | spray=%+.2f m"
          % (r['S_w'], r['b_w']**2/r['S_w'], r['b_w'], r['xi_mlg'], r['spray']))
    print("  SM_aft=%.3f | SM_fwd=%.3f | CLv=%.3f | tank=%.4f | wheel_span=%.2f m"
          % (r['SM_aft'], r['SM_fwd'], r['CLv'], r['tank_excess'],
             r['wheel_span']))
    print()


def _fuse_stations(ap):
    I = ap['inputs']
    L_f, D_f = I['L_f'], I['D_f']
    xts = I['x_tailstrike'] / L_f
    xx = np.array([0.0, 1.24/41.72, 3.54/41.72, 7.55/41.72, xts, 1.0])
    hh = np.array([0.0, 2.27/4.0, 3.56/4.0, 1.0, 1.0, 1.07/4.0]) * D_f
    ww = np.array([0.0, 1.83/4.0, 3.49/4.0, 1.0, 1.0, 0.284/4]) * D_f
    s = np.linspace(0.0, 1.0, 80)
    x = s * L_f
    h = np.interp(s, xx, hh)
    w = np.interp(s, xx, ww)
    return x, h, w


def _draw_case(ax, ap, view, color, ls, lw, label=None, alpha=1.0, fill=None):
    """Contorno 2D de um aviao DT. view: 'top' | 'side' | 'front'."""
    I, G = ap['inputs'], ap['geometry']
    x, h, w = _fuse_stations(ap)
    first = {'label': label}

    def ln(xs, ys, **kw):
        ax.plot(xs, ys, color=color, ls=ls, lw=lw, alpha=alpha, **first, **kw)
        first.pop('label', None)

    def poly(xs, ys, fa=0.18):
        if fill:
            ax.fill(xs, ys, facecolor=fill, edgecolor=color, lw=lw,
                    ls=ls, alpha=fa, zorder=2)

    if view == 'top':
        ln(x, w/2)
        ln(x, -w/2)
        xr, cr, xt, yt, ct = I['xr_w'], G['cr_w'], G['xt_w'], G['yt_w'], G['ct_w']
        wx = [xr, xt, xt + ct, xr + cr, xr]
        wy = [0.0, yt, yt, 0.0, 0.0]
        poly(wx, wy)
        poly(wx, [-v for v in wy])
        ln(wx, wy)
        ln(wx, [-v for v in wy])
        xrh, crh, xth, yth, cth = G['xr_h'], G['cr_h'], G['xt_h'], G['yt_h'], G['ct_h']
        hx = [xrh, xth, xth + cth, xrh + crh, xrh]
        hy = [0.0, yth, yth, 0.0, 0.0]
        ln(hx, hy)
        ln(hx, [-v for v in hy])
        xn, yn, Ln, Dn = I['x_n'], I['y_n'], I['L_n'], I['D_n']
        nx = [xn, xn + Ln, xn + Ln, xn, xn]
        ny = [yn - Dn/2, yn - Dn/2, yn + Dn/2, yn + Dn/2, yn - Dn/2]
        poly(nx, ny, fa=0.35)
        poly(nx, [-v for v in ny], fa=0.35)
        ln(nx, ny)
        ln(nx, [-v for v in ny])
        if I['x_mlg'] is not None:
            ax.plot(I['x_mlg'], I['y_mlg'], 's', color=color, ms=5, alpha=alpha)
            ax.plot(I['x_mlg'], -I['y_mlg'], 's', color=color, ms=5, alpha=alpha)

    elif view == 'side':
        ln(x, h/2)
        ln(x, -h/2)
        xr, cr, zr = I['xr_w'], G['cr_w'], I['zr_w']
        xt, ct, zt = G['xt_w'], G['ct_w'], G['zt_w']
        ln([xr, xt, xt + ct, xr + cr, xr], [zr, zt, zt, zr, zr])
        xrh, crh, zrh = G['xr_h'], G['cr_h'], I['zr_h']
        xth, cth, zth = G['xt_h'], G['ct_h'], G['zt_h']
        ln([xrh, xth, xth + cth, xrh + crh, xrh], [zrh, zth, zth, zrh, zrh])
        xrv, crv, zrv = G['xr_v'], G['cr_v'], I['zr_v']
        xtv, ctv, ztv = G['xt_v'], G['ct_v'], G['zt_v']
        ln([xrv, xtv, xtv + ctv, xrv + crv, xrv], [zrv, ztv, ztv, zrv, zrv])
        xn, zn, Ln, Dn = I['x_n'], I['z_n'], I['L_n'], I['D_n']
        nx = [xn, xn + Ln, xn + Ln, xn, xn]
        nz = [zn - Dn/2, zn - Dn/2, zn + Dn/2, zn + Dn/2, zn - Dn/2]
        poly(nx, nz, fa=0.35)
        ln(nx, nz)
        if I['x_nlg'] is not None:
            ax.plot([I['x_nlg'], I['x_nlg']], [0.0, I['z_lg']],
                    color=color, ls=ls, lw=lw, alpha=alpha)
            ax.plot([I['x_mlg'], I['x_mlg']], [I['zr_w'], I['z_lg']],
                    color=color, ls=ls, lw=lw, alpha=alpha)

    else:
        th = np.linspace(0, 2*np.pi, 80)
        ln(0.5*I['D_f']*np.cos(th), 0.5*I['D_f']*np.sin(th))
        yt, zr, zt = G['yt_w'], I['zr_w'], G['zt_w']
        ln([-yt, 0.0, yt], [zt, zr, zt])
        yth, zrh, zth = G['yt_h'], I['zr_h'], G['zt_h']
        ln([-yth, 0.0, yth], [zth, zrh, zth])
        ln([0.0, 0.0], [I['zr_v'], G['zt_v']])
        yn, zn, Dn = I['y_n'], I['z_n'], I['D_n']
        ln(yn + 0.5*Dn*np.cos(th), zn + 0.5*Dn*np.sin(th))
        ln(-yn + 0.5*Dn*np.cos(th), zn + 0.5*Dn*np.sin(th))
        if I['x_mlg'] is not None:
            ax.plot(I['y_mlg'], I['z_lg'], 's', color=color, ms=5, alpha=alpha)
            ax.plot(-I['y_mlg'], I['z_lg'], 's', color=color, ms=5, alpha=alpha)


def plot_compare(ap_old, ap_new, path=None, r_old=None, r_new=None, show=True):
    """Tres vistas: cinza = atual (my_airplane), azul = nova otimizacao."""
    import matplotlib.pyplot as plt

    if path is None:
        path = os.path.join(RESDIR, "reopt_vs_atual.png")
    g = oc.gravity
    w_old = (r_old or oc.extract(ap_old))['W0'] / g
    w_new = (r_new or oc.extract(ap_new))['W0'] / g
    sm_old = (r_old or oc.extract(ap_old))['SM_aft']
    sm_new = (r_new or oc.extract(ap_new))['SM_aft']

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    views = (
        ('top', 'Superior (x–y)', 'x [m]', 'y [m]'),
        ('side', 'Lateral (x–z)', 'x [m]', 'z [m]'),
        ('front', 'Frontal (y–z)', 'y [m]', 'z [m]'),
    )
    for ax, (view, title, xlab, ylab) in zip(axes, views):
        _draw_case(ax, ap_old, view, color='0.35', ls='--', lw=1.7, fill='0.55',
                   label='atual  W0=%.0f kgf  SM_aft=%.1f%%' % (w_old, 100*sm_old))
        _draw_case(ax, ap_new, view, color='#1f4e8c', ls='-', lw=2.0, fill='#4c7fd0',
                   label='nova   W0=%.0f kgf  SM_aft=%.1f%%' % (w_new, 100*sm_new))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.28)
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    axes[0].legend(loc='upper left', fontsize=8, framealpha=0.92)
    fig.suptitle('Otimizacao (nacele livre) vs configuracao atual', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print("vistas:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


if __name__ == "__main__":
    # Partida = my_airplane (standard_airplane). SLSQP trapezio + nacele livre.
    #   python opt_nac.py              -> otimiza e gera as vistas
    #   python opt_nac.py --plot-only  -> so compara o ultimo JSON com o atual
    import json
    os.makedirs(RESDIR, exist_ok=True)
    json_path = os.path.join(RESDIR, "reopt_nacele_livre.json")
    plot_only = "--plot-only" in sys.argv

    if plot_only:
        if not os.path.isfile(json_path):
            raise SystemExit("nao achei %s — rode python opt_nac.py primeiro" % json_path)
        data = json.load(open(json_path, encoding="utf-8"))
        ap_new = oc.run_designTool(data["inputs"])
        r_new = oc.extract(ap_new)
    else:
        tag = "nacele livre (SLSQP trapezio)"
        model, res, r_new = solve(tag)
        report(tag, model, res, r_new, oc.DV_NAMES)

        def _jsonable(o):
            if isinstance(o, dict):
                return {k: _jsonable(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_jsonable(v) for v in o]
            if isinstance(o, (np.bool_, bool)):
                return bool(o)
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            return o

        out = {
            "success": bool(res.success),
            "message": res.message,
            "n_objfun": model.n_objfun,
            "W0_kgf": float(r_new["W0"] / oc.gravity),
            "x": {n: float(v) for n, v in zip(oc.DV_NAMES, model.to_physical(res.x))},
            "SM_aft": float(r_new["SM_aft"]),
            "SM_fwd": float(r_new["SM_fwd"]),
            "xi_mlg": float(r_new["xi_mlg"]),
            "inputs": _jsonable(model.build_inputs(res.x)),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("gravado:", json_path)
        ap_new = oc.run_designTool(model.build_inputs(res.x))

    ap_old = oc.run_designTool(oc.get_baseline())
    r_old = oc.extract(ap_old)
    print("DT atual (my_airplane)  W0=%.0f kgf  S_w=%.2f  AR=%.3f  xr_w=%.3f  x_n=%.3f  y_n=%.3f  SM_fwd=%.1f%%  SM_aft=%.1f%%"
          % (r_old["W0"] / oc.gravity, r_old["S_w"], r_old["b_w"] ** 2 / r_old["S_w"],
             ap_old["inputs"]["xr_w"], ap_old["inputs"]["x_n"], ap_old["inputs"]["y_n"],
             100.0 * r_old["SM_fwd"], 100.0 * r_old["SM_aft"]))
    plot_compare(ap_old, ap_new, r_old=r_old, r_new=r_new, show=False)

