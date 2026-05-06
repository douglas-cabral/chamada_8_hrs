'''
This script executes the Fokker 100 example
'''

# IMPORTS

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry, change_sweep
from designTool.plots import plot_geometry
from designTool.aerodynamics import aerodynamics
from designTool.weight import weight
from designTool.auxiliary import atmosphere
from designTool.constants import gravity
import pprint

# =========================================

# SETUP

# Constants
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822

# Select airplane name from the standard_airplane function in designTool
# airplane_name = "fokker100"
airplane_name = "my_airplane"

# Polar CL x CD: com accumulate=True, curvas sao somadas entre execucoes via JSON.
PLOT_POLAR_ACCUMULATE = True
PLOT_POLAR_RESET_ACCUMULATOR = False

_POLAR_ACCUMULATOR_PATH = Path(__file__).resolve().parent / "polar_curves_accumulator.json"

# =========================================


def _polar_label(highlift_config):
    label_map = {
        "clean": "Cruzeiro",
        "takeoff": "Decolagem",
        "landing": "Pouso",
        "approach": "Aproximacao",
    }
    return label_map.get(highlift_config, highlift_config)


def compute_polar_curve(
    airplane,
    Mach,
    altitude,
    highlift_config,
    n_engines_failed=0,
    lg_down=0,
    h_ground=0,
    cl_min=-0.5,
    n_points=80,
):
    _, CLmax, _ = aerodynamics(
        airplane,
        Mach,
        altitude,
        0.0,
        n_engines_failed=n_engines_failed,
        highlift_config=highlift_config,
        lg_down=lg_down,
        h_ground=h_ground,
    )
    cl_values = np.linspace(cl_min, CLmax, n_points)
    cd_values = np.zeros_like(cl_values)
    for i, cl in enumerate(cl_values):
        cd_values[i], _, _ = aerodynamics(
            airplane,
            Mach,
            altitude,
            cl,
            n_engines_failed=n_engines_failed,
            highlift_config=highlift_config,
            lg_down=lg_down,
            h_ground=h_ground,
        )
    idx_cd_min = int(np.argmin(cd_values))
    base = _polar_label(highlift_config)
    legend_label = f"{base} (M={Mach:.3f}, h={altitude:.0f} m)"
    return {
        "highlift_config": highlift_config,
        "legend_label": legend_label,
        "Mach": float(Mach),
        "altitude": float(altitude),
        "cl": cl_values.tolist(),
        "cd": cd_values.tolist(),
        "idx_cd_min": idx_cd_min,
    }


def _polar_accumulator_load(path):
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("curves", [])


def _polar_accumulator_save(path, curves):
    path.write_text(json.dumps({"curves": curves}, indent=2), encoding="utf-8")


def plot_polar_curves(curves):
    if not curves:
        return
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(curves):
        color = cmap(i % 10)
        cl = np.asarray(c["cl"], dtype=float)
        cd = np.asarray(c["cd"], dtype=float)
        k = int(c["idx_cd_min"])
        label = c["legend_label"]
        plt.plot(cd, cl, label=label, linewidth=2, color=color)
        plt.scatter(cd[k], cl[k], marker="o", s=70, zorder=3, color=color)
        plt.scatter(cd[0], cl[0], marker="s", s=55, zorder=3, color=color)
        plt.scatter(cd[-1], cl[-1], marker="^", s=65, zorder=3, color=color)
        dy = 16 * i
        plt.annotate(
            f"CD min\nCD={cd[k]:.4f}, CL={cl[k]:.3f}",
            (cd[k], cl[k]),
            textcoords="offset points",
            xytext=(8, 10 + dy),
            fontsize=7,
            color=color,
        )
        plt.annotate(
            f"CL min={cl[0]:.2f}",
            (cd[0], cl[0]),
            textcoords="offset points",
            xytext=(8, -14 - dy),
            fontsize=7,
            color=color,
        )
        plt.annotate(
            f"CL max={cl[-1]:.2f}",
            (cd[-1], cl[-1]),
            textcoords="offset points",
            xytext=(8, 8 + dy),
            fontsize=7,
            color=color,
        )
    plt.xlabel("CD")
    plt.ylabel("CL")
    n = len(curves)
    plt.title(
        "Polar de arrasto (CL x CD)"
        + (f" — {n} curvas" if n > 1 else f" — {curves[0]['legend_label']}")
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cl_vs_cd(
    airplane,
    Mach,
    altitude,
    highlift_config,
    n_engines_failed=0,
    lg_down=0,
    h_ground=0,
    cl_min=-0.5,
    n_points=80,
    accumulate=False,
    reset_accumulator=False,
    accumulator_path=None,
):
    default_path = Path(__file__).resolve().parent / "polar_curves_accumulator.json"
    path = Path(accumulator_path) if accumulator_path else default_path

    if reset_accumulator and path.is_file():
        path.unlink()

    curve = compute_polar_curve(
        airplane,
        Mach,
        altitude,
        highlift_config,
        n_engines_failed=n_engines_failed,
        lg_down=lg_down,
        h_ground=h_ground,
        cl_min=cl_min,
        n_points=n_points,
    )

    if accumulate:
        curves = _polar_accumulator_load(path)
        curves.append(curve)
        _polar_accumulator_save(path, curves)
        plot_polar_curves(curves)
    else:
        plot_polar_curves([curve])


def plot_clmax_vs_sweep_w(
    airplane,
    Mach,
    altitude,
    CL,
    highlift_config,
    n_engines_failed=0,
    lg_down=0,
    h_ground=0,
    sweep_deg_min=0.0,
    sweep_deg_max=60.0,
    n_sweep=61,
    sweep_project_deg=34.0,
):
    """
    CL_max total (retorno de aerodynamics) e CLmax_clean (dragDict) versus enflechamento.
    """
    sweep_orig = airplane["inputs"]["sweep_w"]
    sweep_deg = np.linspace(sweep_deg_min, sweep_deg_max, n_sweep)
    clmax_vals = np.empty_like(sweep_deg, dtype=float)
    clmax_clean_vals = np.empty_like(sweep_deg, dtype=float)

    for i, deg in enumerate(sweep_deg):
        airplane["inputs"]["sweep_w"] = float(np.deg2rad(deg))
        geometry(airplane)
        _, clmax_vals[i], drag_dict = aerodynamics(
            airplane,
            Mach,
            altitude,
            CL,
            n_engines_failed=n_engines_failed,
            highlift_config=highlift_config,
            lg_down=lg_down,
            h_ground=h_ground,
        )
        clmax_clean_vals[i] = float(drag_dict["CLmax_clean"])

    airplane["inputs"]["sweep_w"] = float(np.deg2rad(sweep_project_deg))
    geometry(airplane)
    _, clmax_proj, drag_dict_proj = aerodynamics(
        airplane,
        Mach,
        altitude,
        CL,
        n_engines_failed=n_engines_failed,
        highlift_config=highlift_config,
        lg_down=lg_down,
        h_ground=h_ground,
    )
    clmax_clean_proj = float(drag_dict_proj["CLmax_clean"])

    airplane["inputs"]["sweep_w"] = sweep_orig
    geometry(airplane)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    phase = _polar_label(highlift_config)
    ax.plot(
        sweep_deg,
        clmax_vals,
        "-",
        color="C0",
        linewidth=2,
        label=f"CL$_{{max}}$ ({phase}): clean + flap + slat",
    )
    ax.scatter(
        sweep_project_deg,
        float(clmax_proj),
        s=140,
        zorder=5,
        marker="*",
        color="C3",
        edgecolors="k",
        linewidths=0.6,
        label=f"Projeto ($\\Lambda$ = {sweep_project_deg:.2f}°)",
    )
    ax.axvline(sweep_project_deg, color="C3", linestyle=":", alpha=0.7)
    ax.set_xlabel("Enflechamento da asa $\\Lambda$ [°]")
    ax.set_ylabel("CL$_{max}$ (total)")
    ax.set_title(
        f"CL$_{{max}}$ em funcao do enflechamento ({phase}; "
        r"$C_{L,\max} = C_{L,\max,clean} + \Delta C_{L,\max,flap} + \Delta C_{L,\max,slat}$)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.plot(
        sweep_deg,
        clmax_clean_vals,
        "-",
        color="C2",
        linewidth=2,
        label=r"CL$_{max,clean}$ (asa limpa)",
    )
    ax2.scatter(
        sweep_project_deg,
        clmax_clean_proj,
        s=140,
        zorder=5,
        marker="*",
        color="C3",
        edgecolors="k",
        linewidths=0.6,
        label=f"Projeto ($\\Lambda$ = {sweep_project_deg:.2f}°)",
    )
    ax2.axvline(sweep_project_deg, color="C3", linestyle=":", alpha=0.7)
    ax2.set_xlabel("Enflechamento da asa $\\Lambda$ [°]")
    ax2.set_ylabel(r"CL$_{max,clean}$")
    ax2.set_title(
        r"$C_{L,\max,\mathrm{clean}}$ em funcao do enflechamento da asa"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    plt.show()


# =========================================

# EXECUTION

# Load the airplane dictionary
airplane = standard_airplane(airplane_name)

# Execute the geometry module to compute all dimensions.
# This updates the airplane dictionary with new entries.
geometry(airplane)

# Plot airplane
plot_geometry(airplane, figname="3dview.png", az1=45, az2=-135)

print(pprint.pformat(airplane))

# ---- Weight estimation (converged) ----
W0_guess = airplane['inputs']['W0_guess']
T0_guess = 850000.0
W0, W_empty, W_fuel, W_cruise = weight(W0_guess, T0_guess, airplane)
print(f"\n--- Peso convergido ---")
print(f"W0 (MTOW)  = {W0:.2f} N ({W0/gravity:.0f} kg)")
print(f"W_empty    = {W_empty:.2f} N ({W_empty/gravity:.0f} kg)")
print(f"W_fuel     = {W_fuel:.2f} N ({W_fuel/gravity:.0f} kg)")
print(f"W_cruise   = {W_cruise:.2f} N ({W_cruise/gravity:.0f} kg)")

# ---- Cruise CD vs Mach sweep ----
altitude_cruise = airplane['inputs']['altitude_cruise']
Mach_cruise = airplane['inputs']['Mach_cruise']

atm_cruise = atmosphere(altitude_cruise)
rho_cruise = atm_cruise['density']
a_cruise = atm_cruise['speed_of_sound']

S_w = airplane['inputs']['S_w']
V_cruise = Mach_cruise * a_cruise
q_cruise = 0.5 * rho_cruise * V_cruise**2
CL_cruise = W_cruise / (q_cruise * S_w)

print(f"\nW_cruise  = {W_cruise:.2f} N")
print(f"V_cruise  = {V_cruise:.4f} m/s")
print(f"q_cruise  = {q_cruise:.4f} Pa")
print(f"CL_cruise = {CL_cruise:.6f}")

# Drag breakdown at cruise Mach
CD_cr, CLmax_cr, dd_cr = aerodynamics(
    airplane, Mach_cruise, altitude_cruise, CL_cruise,
    n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0,
)

print("\n" + "=" * 60)
print("  DRAG BREAKDOWN — Cruzeiro (M = {:.2f}, h = 40 000 ft)".format(Mach_cruise))
print("=" * 60)
components = [
    ("CD0_w   (asa)",        dd_cr['CD0_w']),
    ("CD0_h   (HT)",         dd_cr['CD0_h']),
    ("CD0_v   (VT)",         dd_cr['CD0_v']),
    ("CD0_f   (fuselagem)",  dd_cr['CD0_f']),
    ("CD0_n   (nacele)",     dd_cr['CD0_n']),
    ("CD0_flap",             dd_cr['CD0_flap']),
    ("CD0_slat",             dd_cr['CD0_slat']),
    ("CD0_lg  (trem)",       dd_cr['CD0_lg']),
    ("CD0_wdm (windmill)",   dd_cr['CD0_wdm']),
    ("CD0_exc (excresc.)",   dd_cr['CD0_exc']),
]
print(f"  {'Componente':<26s}  {'Valor':>12s}  {'% CD total':>10s}")
print("-" * 60)
for name, val in components:
    pct = val / CD_cr * 100 if CD_cr > 0 else 0.0
    print(f"  {name:<26s}  {val:12.6f}  {pct:9.2f}%")
print("-" * 60)
pct_CD0 = dd_cr['CD0'] / CD_cr * 100
pct_CDind = dd_cr['CDind'] / CD_cr * 100
pct_CDwave = dd_cr['CDwave'] / CD_cr * 100
print(f"  {'CD0 (parasita total)':<26s}  {dd_cr['CD0']:12.6f}  {pct_CD0:9.2f}%")
print(f"  {'CDind (induzido)':<26s}  {dd_cr['CDind']:12.6f}  {pct_CDind:9.2f}%")
print(f"  {'CDwave (onda)':<26s}  {dd_cr['CDwave']:12.6f}  {pct_CDwave:9.2f}%")
print("-" * 60)
print(f"  {'CD TOTAL':<26s}  {CD_cr:12.6f}  {100.0:9.2f}%")
print("=" * 60)

print(f"\n  clmax_w (perfil)  = {airplane['inputs']['clmax_w']:.4f}")
print(f"  CLmax_clean (asa) = {dd_cr['CLmax_clean']:.4f}")
print(f"  CLmax (total)     = {CLmax_cr:.4f}")
print(f"  K (fator induzido)= {dd_cr['K']:.6f}")
print(f"  e (Oswald)        = {dd_cr['e']:.4f}")

LD_cruise = CL_cruise / CD_cr
print(f"\n  L/D (cruzeiro, CL = CL_cruise) = {LD_cruise:.4f}")

_, CLmax_ld, _ = aerodynamics(
    airplane, Mach_cruise, altitude_cruise, 0.0,
    n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0,
)
cl_sweep = np.linspace(0.02, CLmax_ld, 400)
ld_vals = np.empty_like(cl_sweep)
cd_sweep = np.empty_like(cl_sweep)
for i, cl in enumerate(cl_sweep):
    cd_i, _, _ = aerodynamics(
        airplane, Mach_cruise, altitude_cruise, cl,
        n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0,
    )
    cd_sweep[i] = cd_i
    ld_vals[i] = cl / cd_i
k_ld_max = int(np.argmax(ld_vals))
LD_max = ld_vals[k_ld_max]
CL_at_LD_max = cl_sweep[k_ld_max]
CD_at_LD_max = cd_sweep[k_ld_max]
print(f"  (L/D)_max (mesmo M e h)         = {LD_max:.4f}")
print(f"    em CL = {CL_at_LD_max:.6f}, CD = {CD_at_LD_max:.6f}")

# Mach sweep
Mach_range = np.linspace(0.4, 0.9, 100)
CD_values = np.zeros_like(Mach_range)
CDwave_values = np.zeros_like(Mach_range)

for i, M in enumerate(Mach_range):
    CD_val, _, drag_dict = aerodynamics(
        airplane, M, altitude_cruise, CL_cruise,
        n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0,
    )
    CD_values[i] = CD_val
    CDwave_values[i] = drag_dict['CDwave']

# Mach de divergência (equação de Korn)
sweep_w = airplane['inputs']['sweep_w']
k_korn = airplane['inputs']['k_korn']
tcr_w = airplane['inputs']['tcr_w']
tct_w = airplane['inputs']['tct_w']
b_w = airplane['geometry']['b_w']
cr_w = airplane['geometry']['cr_w']
ct_w = airplane['geometry']['ct_w']
tcm_w = 0.25 * tcr_w + 0.75 * tct_w

sweep_50 = change_sweep(0.25, 0.50, sweep_w, b_w / 2, cr_w, ct_w)
Mach_dd = k_korn / np.cos(sweep_50) - tcm_w / np.cos(sweep_50)**2 - CL_cruise / 10 / np.cos(sweep_50)**3
Mach_crit = Mach_dd - (0.1 / 80)**(1 / 3)

print(f"\n  Mach_dd   = {Mach_dd:.6f}")
print(f"  Mach_crit = {Mach_crit:.6f}")

# Plot CD vs Mach
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(Mach_range, CD_values, 'b-', linewidth=2, label='$C_D$ total')
ax1.axvline(Mach_cruise, color='r', linestyle='--', alpha=0.7, label=f'Mach cruzeiro = {Mach_cruise}')
ax1.axvline(Mach_dd, color='green', linestyle='-.', alpha=0.8, label=f'$M_{{dd}}$ = {Mach_dd:.4f}')
ax1.axvline(Mach_crit, color='orange', linestyle=':', alpha=0.8, label=f'$M_{{crit}}$ = {Mach_crit:.4f}')
ax1.set_xlabel('Mach')
ax1.set_ylabel('$C_D$')
ax1.set_title(f'$C_D$ vs Mach (cruzeiro: $C_L$ = {CL_cruise:.4f}, h = 40 000 ft)')
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()

# Plot CDwave vs Mach
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(Mach_range, CDwave_values, 'r-', linewidth=2, label='$C_{D,wave}$')
ax2.axvline(Mach_cruise, color='gray', linestyle='--', alpha=0.7, label=f'Mach cruzeiro = {Mach_cruise}')
ax2.axvline(Mach_dd, color='green', linestyle='-.', alpha=0.8, label=f'$M_{{dd}}$ = {Mach_dd:.4f}')
ax2.axvline(Mach_crit, color='orange', linestyle=':', alpha=0.8, label=f'$M_{{crit}}$ = {Mach_crit:.4f}')
ax2.set_xlabel('Mach')
ax2.set_ylabel('$C_{D,wave}$')
ax2.set_title(f'$C_{{D,wave}}$ vs Mach (cruzeiro: $C_L$ = {CL_cruise:.4f}, h = 40 000 ft)')
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.tight_layout()

# ---- CLmax takeoff vs wing sweep ----
Mach_to = 0.25
altitude_to = airplane['inputs']['altitude_takeoff']
n_engines_failed_to = 1
T0 = T0_guess
d_to = airplane['inputs']['distance_takeoff']

CLmax_to_req = 0.2387 / ((T0 / W0) * d_to) * (W0 / S_w)
print(f"\n  CLmax_to necessario = {CLmax_to_req:.4f}")

sweep_orig = airplane['inputs']['sweep_w']
sweep_deg_arr = np.linspace(0.0, 50.0, 200)
clmax_to_arr = np.empty_like(sweep_deg_arr)

for i, deg in enumerate(sweep_deg_arr):
    airplane['inputs']['sweep_w'] = float(np.deg2rad(deg))
    geometry(airplane)
    _, clmax_to_arr[i], _ = aerodynamics(
        airplane, Mach_to, altitude_to, 0.0,
        n_engines_failed=n_engines_failed_to, highlift_config="takeoff",
        lg_down=0, h_ground=0,
    )

airplane['inputs']['sweep_w'] = sweep_orig
geometry(airplane)

sweep_max_idx = None
if clmax_to_arr[0] >= CLmax_to_req:
    for j in range(len(clmax_to_arr) - 1):
        if clmax_to_arr[j] >= CLmax_to_req and clmax_to_arr[j + 1] < CLmax_to_req:
            frac = (CLmax_to_req - clmax_to_arr[j]) / (clmax_to_arr[j + 1] - clmax_to_arr[j])
            sweep_at_clmax_req = sweep_deg_arr[j] + frac * (sweep_deg_arr[j + 1] - sweep_deg_arr[j])
            sweep_max_idx = j
            break

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(sweep_deg_arr, clmax_to_arr, 'b-', linewidth=2, label=r'$C_{L,\max}$ decolagem')
ax3.axhline(CLmax_to_req, color='r', linestyle='--', alpha=0.7,
            label=f'$C_{{L,\\max,to}}$ necessário = {CLmax_to_req:.4f}')

if sweep_max_idx is not None:
    ax3.axvline(sweep_at_clmax_req, color='green', linestyle='-.', alpha=0.8,
                label=f'$\\Lambda_{{max}}$ = {sweep_at_clmax_req:.2f}°')
    ax3.scatter(sweep_at_clmax_req, CLmax_to_req, s=120, zorder=5, marker='*',
                color='green', edgecolors='k', linewidths=0.6)
    print(f"  Enflechamento maximo para atender CLmax_to: {sweep_at_clmax_req:.2f} deg")

sweep_proj_deg = np.rad2deg(sweep_orig)
airplane['inputs']['sweep_w'] = sweep_orig
geometry(airplane)
_, clmax_proj_to, _ = aerodynamics(
    airplane, Mach_to, altitude_to, 0.0,
    n_engines_failed=n_engines_failed_to, highlift_config="takeoff",
    lg_down=0, h_ground=0,
)
ax3.scatter(sweep_proj_deg, clmax_proj_to, s=140, zorder=5, marker='o',
            color='C3', edgecolors='k', linewidths=0.6,
            label=f'Projeto ($\\Lambda$ = {sweep_proj_deg:.2f}°, $C_{{L,max}}$ = {clmax_proj_to:.4f})')

ax3.set_xlabel('Enflechamento da asa $\\Lambda$ [°]')
ax3.set_ylabel('$C_{L,\\max}$ (decolagem)')
ax3.set_title(
    f'$C_{{L,\\max}}$ de decolagem vs enflechamento '
    f'(M = {Mach_to}, h = 0 ft, 1 motor inop.)'
)
ax3.grid(True, alpha=0.3)
ax3.legend()
fig3.tight_layout()

# ---- Polar CD x CL (cruzeiro e pouso) ----
# (Sem accumulator/JSON: varre CL e chama aerodynamics() ponto a ponto.)
polar_cruise = compute_polar_curve(
    airplane,
    Mach_cruise,
    altitude_cruise,
    highlift_config="clean",
    n_engines_failed=0,
    lg_down=0,
    h_ground=0,
    cl_min=-0.5,
    n_points=400,
)
cl_cruise = np.asarray(polar_cruise["cl"], dtype=float)
cd_cruise = np.asarray(polar_cruise["cd"], dtype=float)

Mach_landing = 0.22
altitude_landing = airplane["inputs"]["altitude_landing"]
h_ground_landing = 0.3 * airplane["geometry"]["b_w"]
polar_landing = compute_polar_curve(
    airplane,
    Mach_landing,
    altitude_landing,
    highlift_config="landing",
    n_engines_failed=0,
    lg_down=1,
    h_ground=h_ground_landing,
    cl_min=-0.5,
    n_points=400,
)
cl_landing = np.asarray(polar_landing["cl"], dtype=float)
cd_landing = np.asarray(polar_landing["cd"], dtype=float)

Mach_second_segment = Mach_to
polar_second_segment = compute_polar_curve(
    airplane,
    Mach_second_segment,
    altitude_to,
    highlift_config="takeoff",
    n_engines_failed=1,
    lg_down=0,
    h_ground=0,
    cl_min=-0.5,
    n_points=400,
)
cl_second_segment = np.asarray(polar_second_segment["cl"], dtype=float)
cd_second_segment = np.asarray(polar_second_segment["cd"], dtype=float)

def max_aerodynamic_efficiency(cl_values, cd_values):
    valid = (cl_values > 0.0) & (cd_values > 0.0)
    ld_values = np.full_like(cl_values, -np.inf, dtype=float)
    ld_values[valid] = cl_values[valid] / cd_values[valid]
    idx = int(np.argmax(ld_values))
    return ld_values[idx], cl_values[idx], cd_values[idx]


efficiency_rows = [
    (
        "Cruzeiro (limpo)",
        Mach_cruise,
        altitude_cruise,
        *max_aerodynamic_efficiency(cl_cruise, cd_cruise),
        LD_cruise,
    ),
    (
        "Pouso",
        Mach_landing,
        altitude_landing,
        *max_aerodynamic_efficiency(cl_landing, cd_landing),
        None,
    ),
    (
        "2o segmento",
        Mach_second_segment,
        altitude_to,
        *max_aerodynamic_efficiency(cl_second_segment, cd_second_segment),
        None,
    ),
]

print("\n" + "=" * 104)
print("  EFICIENCIAS AERODINAMICAS")
print("=" * 104)
print(
    f"  {'Etapa':<22s} {'M':>6s} {'h [m]':>10s} {'(L/D)_max':>12s} "
    f"{'CL@max':>10s} {'CD@max':>10s} {'L/D cruzeiro':>14s}"
)
print("-" * 104)
for stage, mach, altitude, ld_max, cl_at_max, cd_at_max, ld_cr in efficiency_rows:
    ld_cr_text = f"{ld_cr:14.4f}" if ld_cr is not None else f"{'-':>14s}"
    print(
        f"  {stage:<22s} {mach:6.3f} {altitude:10.1f} {ld_max:12.4f} "
        f"{cl_at_max:10.4f} {cd_at_max:10.5f} {ld_cr_text}"
    )
print("=" * 104)

plt.figure(figsize=(10, 7))
plt.plot(
    cd_cruise,
    cl_cruise,
    linewidth=2,
    color="C0",
    label=f"Cruzeiro (M={Mach_cruise:.2f})",
)
plt.plot(
    cd_landing,
    cl_landing,
    linewidth=2,
    color="C1",
    label=f"Pouso (M={Mach_landing:.2f})",
)
plt.plot(
    cd_second_segment,
    cl_second_segment,
    linewidth=2,
    color="C2",
    label=f"2o segmento (M={Mach_second_segment:.2f})",
)
k_cruise = int(polar_cruise["idx_cd_min"])
k_landing = int(polar_landing["idx_cd_min"])
k_second_segment = int(polar_second_segment["idx_cd_min"])
plt.scatter(
    cd_cruise[k_cruise],
    cl_cruise[k_cruise],
    s=70,
    color="C0",
    zorder=3,
)
plt.scatter(
    cd_landing[k_landing],
    cl_landing[k_landing],
    s=70,
    color="C1",
    zorder=3,
)
plt.scatter(
    cd_second_segment[k_second_segment],
    cl_second_segment[k_second_segment],
    s=70,
    color="C2",
    zorder=3,
)
plt.scatter(
    cd_cruise[-1],
    cl_cruise[-1],
    s=90,
    marker="^",
    color="C0",
    zorder=4,
)
plt.scatter(
    cd_landing[-1],
    cl_landing[-1],
    s=90,
    marker="^",
    color="C1",
    zorder=4,
)
plt.scatter(
    cd_second_segment[-1],
    cl_second_segment[-1],
    s=90,
    marker="^",
    color="C2",
    zorder=4,
)
annotation_fontsize = 9
plt.annotate(f"CD={cd_cruise[k_cruise]:.4f}", (cd_cruise[k_cruise], cl_cruise[k_cruise]),
             textcoords="offset points", xytext=(8, -22), fontsize=annotation_fontsize, color="C0")
plt.annotate(f"CD={cd_landing[k_landing]:.4f}", (cd_landing[k_landing], cl_landing[k_landing]),
             textcoords="offset points", xytext=(8, 10), fontsize=annotation_fontsize, color="C1")
plt.annotate(
    f"CD={cd_second_segment[k_second_segment]:.4f}",
    (cd_second_segment[k_second_segment], cl_second_segment[k_second_segment]),
    textcoords="offset points",
    xytext=(8, 14),
    fontsize=annotation_fontsize,
    color="C2",
)
plt.annotate(f"CL={cl_cruise[-1]:.2f}", (cd_cruise[-1], cl_cruise[-1]),
             textcoords="offset points", xytext=(8, 8), fontsize=annotation_fontsize, color="C0")
plt.annotate(f"CL={cl_landing[-1]:.2f}", (cd_landing[-1], cl_landing[-1]),
             textcoords="offset points", xytext=(8, 8), fontsize=annotation_fontsize, color="C1")
plt.annotate(f"CL={cl_second_segment[-1]:.2f}", (cd_second_segment[-1], cl_second_segment[-1]),
             textcoords="offset points", xytext=(8, 8), fontsize=annotation_fontsize, color="C2")
plt.axhline(-0.5, color="k", linestyle="--", linewidth=1, alpha=0.7)
plt.text(plt.xlim()[0], -0.47, "CL = -0.5", fontsize=7, color="k")
plt.xlabel("CD")
plt.ylabel("CL")
plt.title("Polar de arrasto — CL x CD")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

_, CLmax_takeoff_hl, dd_takeoff_hl = aerodynamics(
    airplane,
    Mach_to,
    altitude_to,
    0.0,
    n_engines_failed=n_engines_failed_to,
    highlift_config="takeoff",
    lg_down=0,
    h_ground=0,
)
_, CLmax_landing_hl, dd_landing_hl = aerodynamics(
    airplane,
    Mach_landing,
    altitude_landing,
    0.0,
    n_engines_failed=0,
    highlift_config="landing",
    lg_down=1,
    h_ground=h_ground_landing,
)

print("\n" + "=" * 96)
print("  HIPERSUSTENTADORES E INCREMENTO DE CLmax")
print("=" * 96)
print(f"  Flap escolhido: {airplane['inputs']['flap_type']}")
print(f"  Slat escolhido: {airplane['inputs']['slat_type']}")
print("-" * 96)
print(
    f"  {'Etapa':<12s} {'CLmax clean':>12s} {'Delta flap':>12s} "
    f"{'Delta slat':>12s} {'Delta total':>13s} {'CLmax total':>13s}"
)
print("-" * 96)
for stage, dd_hl, CLmax_hl in [
    ("Decolagem", dd_takeoff_hl, CLmax_takeoff_hl),
    ("Pouso", dd_landing_hl, CLmax_landing_hl),
]:
    delta_flap = dd_hl["deltaCLmax_flap"]
    delta_slat = dd_hl["deltaCLmax_slat"]
    delta_total = delta_flap + delta_slat
    print(
        f"  {stage:<12s} {dd_hl['CLmax_clean']:12.4f} {delta_flap:12.4f} "
        f"{delta_slat:12.4f} {delta_total:13.4f} {CLmax_hl:13.4f}"
    )
print("=" * 96)

plt.show()
