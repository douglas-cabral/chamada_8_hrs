# -*- coding: utf-8 -*-
"""
Re-otimizacao da NJ-0502 com a posicao da nacele (x_n, y_n) como variaveis
de projeto e restricao de water spray. Nao altera os arquivos do Lab 02:
estende DESIGN_VARS / CONSTRAINTS de opt_common em tempo de execucao.
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
    r['x_le_n'] = x_le_n
    # margem do cone de spray: positivo = motor FORA do cone
    r['spray'] = (inp['x_nlg'] + inp['y_n']/np.tan(SPRAY_DEG*deg2rad)
                  - inp['x_n'])
    # geometria util para a quebra
    r['xr_w'] = inp['xr_w']
    r['x_mlg'] = inp['x_mlg']
    r['y_mlg'] = inp['y_mlg']
    r['cr_w'] = geo['cr_w']
    r['ct_w'] = geo['ct_w']
    r['semi_b'] = L
    r['tLE'] = (geo['xt_w'] - inp['xr_w'])/L
    return r


oc.extract = extract

# ---------------------------------------------------------------- restricoes
oc.CONSTRAINTS.append((
    'spray', r'motor fora do cone de spray de $22^\circ$',
    r'$(x_{nlg}+y_n/\tan 22^\circ-x_n)/L_f$',
    lambda r: r['spray']/r['L_f'], 'adicionada'))
oc.CONSTRAINTS.append((
    'nac_inlet', r'entrada da nacele $\geq 0{,}25 L_n$ a frente do BA',
    r'$((x_{LE,n}-x_n)-0{,}25L_n)/L_n$',
    lambda r: ((r['x_le_n'] - r['x_n']) - 0.25*r['L_n'])/r['L_n'], 'adicionada'))
oc.CONSTRAINTS.append((
    'nac_exit', r'saida da nacele $\leq 0{,}25 L_n$ a frente do BA',
    r'$(0{,}25L_n-(x_{LE,n}-x_n-L_n))/L_n$',
    lambda r: (0.25*r['L_n'] - (r['x_le_n'] - r['x_n'] - r['L_n']))/r['L_n'],
    'adicionada'))
oc.CONSTRAINTS.append((
    'nac_gnd', r'nacele $\geq 0{,}5$ m do solo',
    r'$(z_n-D_n/2-z_{lg}-0{,}5)/D_n$',
    lambda r: (r['z_n'] - r['D_n']/2 - r['z_lg'] - 0.5)/r['D_n'], 'adicionada'))


# motor por fora do trem principal: borda interna da nacele >= y_mlg + 1,0 m
oc.CONSTRAINTS.append((
    'nac_mlg', r'nacele por fora do trem ($y_n-D_n/2 \geq y_{mlg}+1$ m)',
    r'$(y_n-D_n/2-y_{mlg}-1)/D_n$',
    lambda r: (r['y_n'] - r['D_n']/2 - r['y_mlg'] - 1.0)/r['D_n'], 'adicionada'))

CRANK_XI = 0.65
CRANK_CB_CT = 1.0      # c_quebra >= CRANK_CB_CT * c_ponta
CRANK_YB_MAX = 0.60    # quebra ate 60% da semi-envergadura (737: 60%)


def _ck(r):
    return crank(r, CRANK_XI)


oc.CONSTRAINTS.append((
    'crank_cb', r'quebra fisica: $c_b \geq k\,c_t$', '',
    lambda r: _ck(r)['c_b']/(CRANK_CB_CT*r['ct_w']) - 1.0, 'adicionada'))
oc.CONSTRAINTS.append((
    'crank_ybmax', r'$y_b \leq 0{,}60\,b/2$', '',
    lambda r: 1.0 - _ck(r)['y_b']/(CRANK_YB_MAX*r['semi_b']), 'adicionada'))
oc.CONSTRAINTS.append((
    'crank_ybmin', r'$y_b \geq y_{mlg}+1$ m', '',
    lambda r: (_ck(r)['y_b'] - r['y_mlg'] - 1.0)/r['semi_b'], 'adicionada'))

oc.CON_NAMES = [c[0] for c in oc.CONSTRAINTS]
oc.CON_INDEX = {c[0]: i for i, c in enumerate(oc.CONSTRAINTS)}


# ------------------------------------------------- quebra (forma fechada)
def crank(r, xi_target=0.65):
    """
    Quebra com BF interno vertical, mantendo S, enflechamento de BA,
    corda de ponta e envergadura. Retorna a quebra necessaria para colocar
    o trem em xi_target da corda local.
    """
    L, t, ct = r['semi_b'], r['tLE'], r['ct_w']
    S2 = r['S_w']/2.0
    x_le_mlg = r['xr_w'] + t*r['y_mlg']
    c_need = (r['x_mlg'] - x_le_mlg)/xi_target      # corda local exigida
    c_root = c_need + t*r['y_mlg']                  # BF vertical -> raiz
    den = (c_root - ct - t*L)/2.0
    y_b = (S2 - L*(c_root + ct)/2.0)/den if abs(den) > 1e-12 else np.nan
    c_b = c_root - t*y_b
    return dict(c_need=c_need, c_root=c_root, y_b=y_b, c_b=c_b,
                x_te=r['xr_w'] + c_root,
                ok=(c_b >= ct) and (y_b > r['y_mlg']) and (y_b < L))


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
    c = crank(r, 0.65)
    print("  QUEBRA para xi=0.65: c_raiz %.3f -> %.3f m | y_b = %.2f m (%.0f%% semi-env.)"
          % (r['cr_w'], c['c_root'], c['y_b'], 100*c['y_b']/r['semi_b']))
    print("                       c_quebra = %.3f m (c_ponta = %.3f m) -> %s"
          % (c['c_b'], r['ct_w'], "VIAVEL" if c['ok'] else "INVIAVEL"))
    print()
    return c
