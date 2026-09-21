# -*- coding: utf-8 -*-
import json
import os as _os
import sys as _s
_s.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import subprocess
from avl_check import RESDIR
import numpy as np
import opt_avl as oa, opt_nac as on, opt_common as oc, avl_check as ac

d = json.load(open(_os.path.join(RESDIR, 'final_design.json'))); inp = d['inputs']
oa.FUEL_CREDIT = False
ap = oc.run_designTool(inp); I, G = ap['inputs'], ap['geometry']; L = G['b_w']/2
P = ac.planform(ap, d['y_b']); P['Y_B'] = d['y_b']
P0 = ac.planform(ap, 1e-6)
dcg = ap['empty_weight']['W_w']*(P['xcg_w'] - P0['xcg_w'])/ap['thrust_matching']['W0']
xa = ap['balance']['xcg_aft'] + dcg
yf = I['b_flap_b_wing']*L
hdr = ["Unidades: METROS / graus. Otimo re-obtido (SLSQP) com nacele livre + water spray,",
       "margem estatica verificada no AVL e quebra tipo Yehudi no bordo de fuga interno.",
       "BF interno VERTICAL em x=%.3f m, quebra em y=%.3f m (%.1f%% b/2); trem principal a 70%% da corda local."
       % (I['xr_w'] + d['c_root'], d['y_b'], 100*d['y_b']/L),
       "S=%.3f m2, b=%.3f m, corda de ponta=%.4f m (quebra preserva S, BA, c_t e b da asa otimizada)."
       % (P['S'], 2*L, P['ct']),
       "Flap c_f/c=0.32 do lado da fuselagem ate y=%.2f m (66%% b/2); nacele x_n=%.3f y_n=%.3f z_n=%.3f."
       % (yf, I['x_n'], I['y_n'], I['z_n']),
       "Xref = CG traseiro corrigido pela quebra. Perfil a1.dat e washout linear de -3 deg (hipoteses)."]
path = os.path.join(ac.AVLDIR, 'nj0502_otim_quebra.avl')
ac.write_avl(ap, P, xa, path, yf=yf, header=hdr)
X = ac.xnp('nj0502_otim_quebra.avl')
print("Xnp = %.4f m | xcg_aft* = %.4f | MAC = %.4f | SM_aft = %.2f%%" % (X, xa, P['MAC'], 100*(X - xa)/P['MAC']))
out = subprocess.run([os.path.join(ac.AVLDIR, 'avl337.exe')],
                     input="load nj0502_otim_quebra.avl\noper\nx\n\nquit\n", capture_output=True,
                     text=True, cwd=ac.AVLDIR, timeout=300).stdout
print("\n".join(l for l in out.splitlines() if "Building" in l or "***" in l or "D1 " in l or "D2 " in l
                or "D3 " in l or "D4 " in l or "D5 " in l))

# ---- novos inputs para 'my_airplane' (designTool/standard_airplane.py) ----
keys = ['S_w', 'AR_w', 'sweep_w', 'xr_w', 'Cht', 'Lc_h', 'Cvt', 'Lb_v',
        'x_mlg', 'y_mlg', 'z_lg', 'x_n', 'y_n', 'z_n']
L_ = ["# NJ-0502 - novo otimo (nacele livre + water spray + SM verificada no AVL + quebra Yehudi)",
      "# Substitui as entradas correspondentes de 'my_airplane' em designTool/standard_airplane.py.",
      "# A quebra NAO e modelada pelo designTool: a geometria dela esta em nj0502_otim_quebra.avl.",
      "# Gerado por estudo_quebra_asa/scripts/03_gera_avl_final.py.",
      "import numpy as np", "", "novos_inputs = {"]
for k in keys:
    if k == 'sweep_w':
        L_.append("    'sweep_w' : %.10f*np.pi/180," % np.degrees(inp[k]))
    else:
        L_.append("    %-9s: %.10f," % ("'%s'" % k, inp[k]))
L_ += ["}", "", "# Quebra (BF interno vertical): c_raiz = %.4f m, x_BF = %.4f m, y_quebra = %.4f m"
       % (d['c_root'], inp['xr_w'] + d['c_root'], d['y_b'])]
open(os.path.join(ac.AVLDIR, 'nj0502_otim_quebra_inputs.py'), 'w', encoding='utf-8').write("\n".join(L_) + "\n")
print("inputs gravados em nj0502_otim_quebra_inputs.py")
