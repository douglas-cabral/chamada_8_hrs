# Sample script on how to use the thrust_matching function from designTool.
# Remember to save this script in the same directory as designTool.py

# IMPORTS
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
from designTool.constants import gravity
import numpy as np
import matplotlib.pyplot as plt

# True: varredura com peso do motor (weight) e marca ponto de projeto (Tmax).
# False: varredura “livre” — remove Tmax e weight do engine; T0 = 1,05·max(T0req);
#         chutes reiniciados a cada S_w (evita propagar NaN em S_w pequenos).
SHOW_DESIGN_POINT = True


def _interp_sw_delta_zero(sw, delta):
    """S_w onde deltaS_wlan = 0 (interpolação linear entre vizinhos)."""
    for i in range(len(delta) - 1):
        d0, d1 = float(delta[i]), float(delta[i + 1])
        x0, x1 = float(sw[i]), float(sw[i + 1])
        if np.isnan(d0) or np.isnan(d1):
            continue
        if d0 == 0.0:
            return x0
        if d0 * d1 < 0.0:
            return x0 - d0 * (x1 - x0) / (d1 - d0)
    return None


def _interp_sw_y_cross(sw, y, y_target):
    """S_w onde y(S) = y_target (interpolação linear entre vizinhos)."""
    sw = np.asarray(sw, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        d0, d1 = y[i] - y_target, y[i + 1] - y_target
        if np.isnan(d0) or np.isnan(d1):
            continue
        if d0 == 0.0:
            return float(sw[i])
        if d0 * d1 < 0.0:
            return float(sw[i] - d0 * (sw[i + 1] - sw[i]) / (d1 - d0))
    return None

# Load a sample case already defined in designTools.py:
airplane = standard_airplane('my_airplane')

# S_w do ponto de projeto (standard_airplane); a varredura usa outros S_w temporariamente
S_w_design = airplane['inputs']['S_w']

# Execute the geometry function
geometry(airplane)

# Chutes iniciais (reutilizados em cada S_w se SHOW_DESIGN_POINT for False)
W0_init = airplane['inputs']['W0_guess']
T0_init = 862000
W0_guess = W0_init
T0_guess = T0_init

# 3

S_min = 0.65 * airplane['inputs']['S_w']
S_max = 1.3 * airplane['inputs']['S_w']

S_array = np.linspace(S_min, S_max, 100)

# MTOW vs S_w, T_i vs S_w
MTOW_array = []
T0_array = []
delta_lan_array = []
T0req_arrays = {
    'Takeoff': [],
    'Cruise': [],
    'High speed cruise': [],
    'FAR 25.111': [],
    'FAR 25.121a': [],
    'FAR 25.121b': [],
    'FAR 25.121c': [],
    'FAR 25.119': [],
    'FAR 25.121d': []
}

# Com 'Tmax' no engine, thrust_matching fixa T0 = n_engines*Tmax (constante).
# Na varredura remove-se Tmax para obter T0 = 1,05*max(T0req) em cada S_w.
_engine = airplane['inputs']['engine']
_tmax_sl = _engine.pop('Tmax', None)
_weight_sl = None
if not SHOW_DESIGN_POINT:
    _weight_sl = _engine.pop('weight', None)

for S_w in S_array:
    airplane['inputs']['S_w'] = S_w
    geometry(airplane)

    if SHOW_DESIGN_POINT:
        w0_try, t0_try = W0_guess, T0_guess
    else:
        w0_try, t0_try = W0_init, T0_init

    thrust_matching(w0_try, t0_try, airplane)
    tm = airplane['thrust_matching']

    if SHOW_DESIGN_POINT:
        W0_guess = tm['W0']
        T0_guess = tm['T0']

    w0_pt = float(tm['W0'])
    t0req_vals = list(tm['T0req'].values())
    t0_pt = 1.05 * max(t0req_vals) if t0req_vals else np.nan
    delta_pt = float(tm['deltaS_wlan'])

    if not np.isfinite(w0_pt) or not np.isfinite(t0_pt):
        w0_pt = np.nan
        t0_pt = np.nan
        delta_pt = np.nan
        t0req_pt = {key: np.nan for key in T0req_arrays}
    else:
        t0req_pt = tm['T0req']

    MTOW_array.append(w0_pt)
    T0_array.append(t0_pt)
    delta_lan_array.append(delta_pt)
    for key in T0req_arrays.keys():
        T0req_arrays[key].append(t0req_pt[key])

if _tmax_sl is not None:
    _engine['Tmax'] = _tmax_sl
if _weight_sl is not None:
    _engine['weight'] = _weight_sl

_delta = np.asarray(delta_lan_array, dtype=float)
_mask = np.isfinite(_delta)
sw_landing = _interp_sw_delta_zero(S_array[_mask], _delta[_mask])
_vline_label = r"Limite de pouso ($\Delta S_{\mathrm{wlan}} = 0$)"
if sw_landing is not None:
    print(f"S_w onde deltaS_wlan = 0: {sw_landing:.2f} m²")

S_array = np.asarray(S_array, dtype=float)
MTOW_array = np.asarray(MTOW_array, dtype=float)
T0_array = np.asarray(T0_array, dtype=float)

# Empuxo máximo instalado (Tmax por motor × nº de motores)
n_engines = airplane['inputs']['n_engines']
T0_motor = n_engines * _tmax_sl if _tmax_sl is not None else None

# Ponto de projeto: S_w fixo no dicionário; MTOW por thrust_matching nessa área
S_design = None
W0_design = None
if SHOW_DESIGN_POINT and T0_motor is not None:
    S_design = float(S_w_design)
    airplane['inputs']['S_w'] = S_design
    geometry(airplane)
    thrust_matching(
        airplane['inputs']['W0_guess'],
        T0_motor,
        airplane,
    )
    W0_design = float(airplane['thrust_matching']['W0'])
    print(
        f"Ponto de projeto: S_w = {S_design:.2f} m² (standard_airplane), "
        f"T0 = {T0_motor/1000:.1f} kN (Tmax motor), "
        f"MTOW = {W0_design/gravity:.0f} kg"
    )

# Plot MTOW vs S_W
plt.plot(S_array, MTOW_array, label="MTOW convergido")
if sw_landing is not None:
    plt.axvline(sw_landing, color="C3", linestyle="--", lw=2, label=_vline_label)
if SHOW_DESIGN_POINT and S_design is not None:
    plt.scatter(
        [S_design], [W0_design], s=160, marker="*", color="C2", edgecolors="k",
        linewidths=0.8, zorder=6, label="Ponto de projeto",
    )
    plt.annotate(
        f"$S_w$ = {S_design:.1f} m²\nMTOW = {W0_design/gravity:.0f} kg",
        (S_design, W0_design),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )
plt.xlabel('S_w')
plt.ylabel('MTOW')
plt.title('MTOW vs S_w')
plt.legend(loc="best")
plt.show()

# Plot T_i vs S_w
fig_t0, ax_t0 = plt.subplots(figsize=(12, 7))
for key in T0req_arrays.keys():
    ax_t0.plot(S_array, T0req_arrays[key], label=key, zorder=2)

ax_t0.plot(
    S_array,
    T0_array,
    "k-",
    lw=2.5,
    label=r"$T_0 = 1{,}05 \times \max(T_{0,\mathrm{req}})$",
    zorder=3,
)

if sw_landing is not None:
    ax_t0.axvline(sw_landing, color="C3", linestyle="--", lw=2, label=_vline_label, zorder=4)
if SHOW_DESIGN_POINT and T0_motor is not None:
    ax_t0.axhline(
        T0_motor, color="gray", linestyle=":", lw=1.5, alpha=0.8,
        label=r"$T_{0,\mathrm{motor}}$ (Tmax × $n_{\mathrm{eng}}$)",
        zorder=4,
    )
if SHOW_DESIGN_POINT and S_design is not None and T0_motor is not None:
    ax_t0.scatter(
        [S_design], [T0_motor], s=160, marker="*", color="C2", edgecolors="k",
        linewidths=0.8, zorder=20, label="Ponto de projeto",
    )

ax_t0.set_xlabel('S_w')
ax_t0.set_ylabel('T0 [N]')
ax_t0.set_title('T0 requerido vs S_w')
ax_t0.grid(True, alpha=0.3)
leg_t0 = ax_t0.legend(loc="best", fontsize=7)
leg_t0.set_zorder(5)

# Caixa de texto por último, à frente das curvas e da legenda
if SHOW_DESIGN_POINT and S_design is not None and T0_motor is not None:
    ann_t0 = ax_t0.annotate(
        f"$S_w$ = {S_design:.1f} m²\n$T_0$ = {T0_motor/1000:.1f} kN",
        (S_design, T0_motor),
        textcoords="offset points",
        xytext=(12, -28),
        fontsize=9,
        clip_on=False,
        zorder=25,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=1.0),
    )
    ann_t0.set_zorder(25)

fig_t0.tight_layout()
plt.show()
