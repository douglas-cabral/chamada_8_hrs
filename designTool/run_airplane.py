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
import pprint

# =========================================

# SETUP

# Constants
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
gravity = 9.81

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
    path = Path(accumulator_path) if accumulator_path else _POLAR_ACCUMULATOR_PATH

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
    sweep_project_deg=31.91,
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
        r"$C_{L,\max,\mathrm{clean}}$ em funcao do enflechamento da asa (asa limpa)"
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

# ---- Cruise CD vs Mach sweep ----
W0 = 280000 * gravity
W_cruise = 0.96 * W0
altitude_cruise = 12192.0  # 40000 ft in meters
rho_cruise = 0.30267       # kg/m^3
a_cruise = 295.0695        # m/s
Mach_cruise = 0.85

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
Mach_to = 0.3
altitude_to = 0.0
n_engines_failed_to = 1
T0 = 748840.0       # N
d_to = 2900.0        # m

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

plt.show()
