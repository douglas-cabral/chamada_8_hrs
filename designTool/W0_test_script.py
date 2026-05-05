# Sample script on how to use the weight function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.weight import weight
from designTool.constants import gravity
import numpy as np
import pprint

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# Execute the geometry function
geometry(airplane)

# Guess values for initial iteration
W0_guess = 467500.00000000000000
T0_guess = 140250.00000000000000

# Execute the weight estimation
W0, W_empty, W_fuel, W_cruise = weight(W0_guess, T0_guess, airplane)

# Print results: pesos em N e kgf arredondados (inteiros), % do W0 inteiro
def _int_w(x):
    return int(round(float(x)))


def _pct_w0_str(w_part, w0_ref):
    return f"{100.0 * float(w_part) / float(w0_ref):.1f}"


def _fmt_N_kgf_pct(W, W0_ref):
    fw = float(W)
    w0 = float(W0_ref)
    w_n = _int_w(fw)
    w_kgf = _int_w(fw / gravity)
    return f"{w_n} N  ({w_kgf} kgf)  [{_pct_w0_str(fw, w0)}% do W0]"


print("W0 = ", _fmt_N_kgf_pct(W0, W0))
print("W_empty = ", _fmt_N_kgf_pct(W_empty, W0))
print("W_fuel = ", _fmt_N_kgf_pct(W_fuel, W0))
print("W_cruise = ", _fmt_N_kgf_pct(W_cruise, W0))
print()
wempty_pesos_int = {
    k: (_int_w(v) if k.startswith("W_") else v)
    for k, v in airplane["empty_weight"].items()
}
print("Wempty_dict = " + pprint.pformat(wempty_pesos_int))
w0f = float(W0)
empty_w_kgf_pct = {
    k: f"{_int_w(float(v) / gravity)} kgf  [{_pct_w0_str(v, w0f)}% do W0]"
    for k, v in airplane["empty_weight"].items()
    if k.startswith("W_")
}
print("Wempty_dict (componentes em kgf e % do W0) = " + pprint.pformat(empty_w_kgf_pct))
print()
fuel_w = airplane["fuel_weight"]
# C_hist no modulo: 1/s; no terminal tambem 1/h (x 3600)
c_hist_1_per_s = {k: float(v) for k, v in fuel_w["C_hist"].items()}
wfuel_display = {
    "Mf_hist": fuel_w["Mf_hist"],
    "LD_hist": fuel_w["LD_hist"],
    "C_hist (1/s)": c_hist_1_per_s,
    "C_hist (1/h)": {k: v * 3600.0 for k, v in c_hist_1_per_s.items()},
    "trapped_fuel_factor": fuel_w["trapped_fuel_factor"],
}
print(
    "Wfuel_dict (C_hist original 1/s e equivalente 1/h) = "
    + pprint.pformat(wfuel_display)
)
print()

# Consumo de combustível por etapa a partir de Mf_hist (mesma ordem que em weight.py)
MF_ORDER = (
    "start",
    "taxi",
    "takeoff",
    "climb",
    "cruise",
    "descent",
    "altcruise",
    "loiter",
    "landing",
)


def combustivel_por_etapa_kgf(W0_N, Mf_hist):
    """
    Em cada etapa k, Mf_k = W_fim_k / W_início_k (fração de peso remanescente).

    P_prev = produto dos Mf de todas as etapas anteriores a k (1 se k é a primeira).
    W_início_k = W0 * P_prev  [N]
    ΔW_k = W_início_k * (1 - Mf_k)  [N]  (peso de combustível consumido na etapa)
    kgf_k = ΔW_k / g
    """
    rows = []
    product_prev = 1.0
    w0 = float(W0_N)
    for name in MF_ORDER:
        mf = float(Mf_hist[name])
        w_inicio = w0 * product_prev
        delta_w = w_inicio * (1.0 - mf)
        kgf = delta_w / gravity
        rows.append(
            {
                "etapa": name,
                "Mf_k": mf,
                "P_prev": product_prev,
                "W_inicio_N": w_inicio,
                "delta_W_N": delta_w,
                "delta_kgf": kgf,
            }
        )
        product_prev *= mf
    return rows


# Missao com MTOW fixo (kgf -> N) e Mf iguais à referencia, exceto cruzeiro = 0.9532
W0_MTOW_KGF = 41565
W0_mission = float(W0_MTOW_KGF) * gravity
mf_ref = {k: float(airplane["fuel_weight"]["Mf_hist"][k]) for k in MF_ORDER}
mf_mission = {**mf_ref, "cruise": 0.9532}
rows_mf = combustivel_por_etapa_kgf(W0_mission, mf_mission)
w0_float = W0_mission
print(
    f"Consumo por etapa (MTOW = {W0_MTOW_KGF} kgf = {_int_w(W0_mission)} N; "
    "Mf por etapa iguais à missao de referencia do weight(), cruzeiro = 0.9532):"
)
print("Mf_mission = " + pprint.pformat(mf_mission))
print(
    "(Soma dW das etapas = W0*(1-prod(Mf)). Combustivel preso = 6% dessa soma.)\n"
)
for r in rows_mf:
    d_w = r["delta_W_N"]
    print(
        f"  {r['etapa']:<10}  Mf={r['Mf_k']:.6f}  "
        f"P_prev={r['P_prev']:.6f}  W_inicio={_int_w(r['W_inicio_N'])} N  "
        f"dW={_int_w(d_w)} N  ({_int_w(r['delta_kgf'])} kgf)  "
        f"[{_pct_w0_str(d_w, w0_float)}% do W0]"
    )
soma_delta = sum(r["delta_W_N"] for r in rows_mf)
mf_total = float(np.prod([float(mf_mission[k]) for k in MF_ORDER]))
trapped_frac = 0.06
trapped_N = trapped_frac * soma_delta
print()
print(
    f"  Soma dW (etapas) = {_int_w(soma_delta)} N  "
    f"({_int_w(soma_delta / gravity)} kgf)  "
    f"[{_pct_w0_str(soma_delta, w0_float)}% do W0]"
)
print(
    f"  W0*(1-prod(Mf))  = {_int_w(w0_float * (1.0 - mf_total))} N  (checagem)"
)
print(
    f"  Combustivel preso (6% da soma dW) = {_int_w(trapped_N)} N  "
    f"({_int_w(trapped_N / gravity)} kgf)  "
    f"[{_pct_w0_str(trapped_N, w0_float)}% do W0]"
)
print(
    f"  Total combustivel (etapas + preso) = {_int_w(soma_delta + trapped_N)} N  "
    f"({_int_w((soma_delta + trapped_N) / gravity)} kgf)  "
    f"[{_pct_w0_str(soma_delta + trapped_N, w0_float)}% do W0]"
)
