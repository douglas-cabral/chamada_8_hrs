"""
Diagrama de projeto (matching chart) com as etapas:
- Decolagem
- Subida
- Cruzeiro
- Pouso

Preencha os valores no dicionario INPUTS e execute:
    python Diagrama_projeto.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from designTool.standard_airplane import standard_airplane
from designTool.geometry import geometry
from designTool.performance import thrust_matching
from designTool.weight import weight
from designTool.aerodynamics import aerodynamics
from designTool.auxiliary import atmosphere
from designTool.propulsion import engineTSFC
from designTool.constants import gravity


# ============================================================================
# ENTRADAS — calculadas automaticamente a partir do designTool
# ============================================================================
# Avaliado no mesmo avião (standard_airplane) e nas mesmas condições de
# performance.py / weight() para evitar valores desatualizados.
AIRPLANE_NAME = "my_airplane"

# Constantes de plotagem e do modelo de pouso (Torenbeek), independentes do avião.
PLOT_AND_LANDING_CONSTANTS = {
    "ws_min": 1500,
    "ws_max": 14200,
    "n_points": 400,
    "g": gravity,           # [m/s^2] gravidade (constante do designTool)
    "a_g": 0.5,             # [-] desaceleracao media (Torenbeek, performance.py)
    "f_ld": 5.0 / 3.0,      # [-] fator de seguranca FAR (performance.py)
    "h_ld": 15.3,           # [m] altura de obstaculo no pouso (performance.py)
}


def build_inputs_from_designtool(name: str = AIRPLANE_NAME) -> dict:
    """Monta o dicionario INPUTS lendo o avião do designTool.

    Todos os valores derivam de standard_airplane(name) + geometry() +
    thrust_matching() + weight(), avaliados nas mesmas condicoes usadas em
    performance.py (decolagem, FAR 25.121b, cruzeiro e pouso).
    """
    airplane = standard_airplane(name)
    geometry(airplane)
    inp = airplane["inputs"]

    n_eng = inp["n_engines"]
    S_w = inp["S_w"]
    W0_guess = inp["W0_guess"]
    T0_inst = n_eng * inp["engine"]["Tmax"]  # tracao instalada (2 x Tmax)

    # Peso convergido + desempenho (mesmo fluxo do designTool)
    thrust_matching(W0_guess, T0_inst, airplane)
    tm = airplane["thrust_matching"]
    W0 = tm["W0"]
    T0 = tm["T0"]
    CLmaxTO = tm["CLmaxTO"]
    _, _, _, W_cruise = weight(W0_guess, T0_inst, airplane)

    # --- Decolagem ---
    atm_to = atmosphere(inp["altitude_takeoff"], inp["deltaISA_takeoff"])
    rho_to = atm_to["density"]

    # --- Subida FAR 25.121b (2o segmento, OEI) ---
    ks = 1.2
    if n_eng <= 2:
        gamma_cl = 0.024
    elif n_eng == 3:
        gamma_cl = 0.027
    else:
        gamma_cl = 0.030
    CL_cl = CLmaxTO / ks**2
    V_cl = np.sqrt(2 * W0 / rho_to / S_w / CL_cl)
    Mach_cl = V_cl / atm_to["speed_of_sound"]
    _, _, dd_cl = aerodynamics(
        airplane, Mach=Mach_cl, altitude=inp["altitude_takeoff"], CL=CL_cl,
        n_engines_failed=1, highlift_config="takeoff", lg_down=0, h_ground=0,
    )

    # --- Cruzeiro ---
    atm_cr = atmosphere(inp["altitude_cruise"])
    rho_cr = atm_cr["density"]
    V_cr = inp["Mach_cruise"] * atm_cr["speed_of_sound"]
    CL_cr = 2.0 * W_cruise / rho_cr / S_w / V_cr**2
    _, _, dd_cr = aerodynamics(
        airplane, Mach=inp["Mach_cruise"], altitude=inp["altitude_cruise"], CL=CL_cr,
        n_engines_failed=0, highlift_config="clean", lg_down=0, h_ground=0,
    )
    _, kT_cr = engineTSFC(inp["Mach_cruise"], inp["altitude_cruise"], airplane)

    # --- Pouso ---
    atm_ld = atmosphere(inp["altitude_landing"], inp["deltaISA_landing"])
    rho_ld = atm_ld["density"]
    _, CLmaxLD, _ = aerodynamics(
        airplane, Mach=0.2, altitude=inp["altitude_landing"], CL=0.5,
        n_engines_failed=0, highlift_config="landing", lg_down=1,
        h_ground=inp["h_ground"],
    )

    data = dict(PLOT_AND_LANDING_CONSTANTS)
    data.update({
        # Ponto de projeto
        "ws_project": W0 / S_w,
        "tw_project": T0 / W0,
        "w0": W0,
        "t0_fixed": T0,
        # Decolagem
        "rho_to": rho_to,
        "clmax_to": CLmaxTO,
        "d_to": inp["distance_takeoff"],
        # Subida (FAR 25.121b)
        "ks": ks,
        "gamma_cl": gamma_cl,
        "n_eng": n_eng,
        "n_eng_failed": 1,
        "cd0_cl": dd_cl["CD0"],
        "k_cl": dd_cl["K"],
        "clmax_cl": CLmaxTO,
        "wcl_w0": 1.0,
        "tcl_t0": 1.0,
        # Cruzeiro
        "rho_cruise": rho_cr,
        "v_cruise": V_cr,
        "cd0_cr": dd_cr["CD0"],
        "k_cr": dd_cr["K"],
        "wcr_w0": W_cruise / W0,
        "tcr_t0": kT_cr,
        # Pouso
        "rho_landing": rho_ld,
        "clmax_landing": CLmaxLD,
        "d_landing": inp["distance_landing"],
        "wld_w0": inp["MLW_frac"],
    })
    return data


INPUTS = build_inputs_from_designtool()


def _require(data: dict, keys: list[str]) -> None:
    """Lanca erro amigavel caso algum dado obrigatorio esteja vazio."""
    missing = [k for k in keys if data.get(k) is None]
    if missing:
        msg = "\n".join(f"- {k}" for k in missing)
        raise ValueError(
            "Preencha os seguintes campos em INPUTS antes de plotar:\n" + msg
        )


def takeoff_constraint(ws: np.ndarray, data: dict) -> np.ndarray:
    """(T0/W0)_TO = 0.2387/(sigma*CLmax_TO*d_TO) * (W0/S)."""
    _require(data, ["rho_to", "clmax_to", "d_to"])
    sigma = data["rho_to"] / 1.225
    coef = 0.2387 / (sigma * data["clmax_to"] * data["d_to"])
    return coef * ws


def climb_constraint(ws: np.ndarray, data: dict) -> np.ndarray:
    """
    Subida:
    (T/W)_CL = ks^2/CLmax_CL * CD0_CL + CLmax_CL/ks^2 * K_CL + gamma
    (T0/W0)_CL = (W_CL/W0)/(T_CL/T0) * N_eng/(N_eng-N_eng_f) * (T/W)_CL
    """
    _require(
        data,
        [
            "ks",
            "clmax_cl",
            "cd0_cl",
            "k_cl",
            "gamma_cl",
            "wcl_w0",
            "tcl_t0",
            "n_eng",
            "n_eng_failed",
        ],
    )
    ks = data["ks"]
    tw_cl = (
        (ks**2 / data["clmax_cl"]) * data["cd0_cl"]
        + (data["clmax_cl"] / ks**2) * data["k_cl"]
        + data["gamma_cl"]
    )

    if data["n_eng_failed"] >= data["n_eng"]:
        raise ValueError("n_eng_failed deve ser menor que n_eng na subida.")
    if data["tcl_t0"] <= 0.0:
        raise ValueError("tcl_t0 deve ser maior que zero na subida.")

    # Conversao de requisito de subida para T0/W0:
    # 1) Correcao de peso: W_CL/W0
    # 2) Correcao de tracao: 1/(T_CL/T0)
    # 3) Penalidade de motor inoperante: N_eng/(N_eng - N_eng_failed)
    weight_factor = data["wcl_w0"]
    thrust_factor = 1.0 / data["tcl_t0"]
    engine_factor = data["n_eng"] / (data["n_eng"] - data["n_eng_failed"])

    tw_to = weight_factor * thrust_factor * engine_factor * tw_cl
    return np.full_like(ws, tw_to, dtype=float)


def cruise_constraint(ws: np.ndarray, data: dict) -> np.ndarray:
    """
    Cruzeiro:
    (W/S)_CR = (W_CR/W0) * (W0/S)
    (T/W)_CR = q_CR/(W/S)_CR * CD0_CR + (W/S)_CR/q_CR * K_CR
    (T0/W0)_CR = (W_CR/W0)/(T_CR/T0) * (T/W)_CR
    """
    _require(data, ["rho_cruise", "v_cruise", "cd0_cr", "k_cr", "wcr_w0", "tcr_t0"])
    q_cr = 0.5 * data["rho_cruise"] * data["v_cruise"] ** 2
    ws_cr = data["wcr_w0"] * ws

    tw_cr = q_cr / ws_cr * data["cd0_cr"] + ws_cr / q_cr * data["k_cr"]
    tw_to = (data["wcr_w0"] / data["tcr_t0"]) * tw_cr
    return tw_to


def landing_wing_loading_limit(data: dict) -> float:
    """
    Pouso:
    x_LD = 1.52/a_g + 1.69
    A_LD = g/(f_LD*x_LD)
    B_LD = -10*g*(h_LD/x_LD)
    (W/S)_LD = rho_LD * CLmax_LD * (A_LD*d_LD + B_LD)
    (W0/S)_LD = 1/(W_LD/W0) * (W/S)_LD
    """
    _require(
        data,
        ["a_g", "g", "f_ld", "h_ld", "rho_landing", "clmax_landing", "d_landing", "wld_w0"],
    )

    x_ld = 1.52 / data["a_g"] + 1.69
    a_ld = data["g"] / (data["f_ld"] * x_ld)
    b_ld = -10.0 * data["g"] * (data["h_ld"] / x_ld)
    ws_ld = data["rho_landing"] * data["clmax_landing"] * (a_ld * data["d_landing"] + b_ld)
    w0s_limit = ws_ld / data["wld_w0"]
    return w0s_limit


def main() -> None:
    ws_landing_limit = landing_wing_loading_limit(INPUTS)
    ws_project = INPUTS["ws_project"]
    tw_project = INPUTS["tw_project"]

    # Todas as curvas usam exatamente o range definido em INPUTS.
    ws_min_base = INPUTS["ws_min"]
    ws_max_base = INPUTS["ws_max"]
    ws = np.linspace(ws_min_base, ws_max_base, int(INPUTS["n_points"]))

    tw_takeoff = takeoff_constraint(ws, INPUTS)
    tw_climb = climb_constraint(ws, INPUTS)
    tw_cruise = cruise_constraint(ws, INPUTS)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ws, tw_takeoff, lw=2.0, label="Decolagem")
    ax.plot(ws, tw_climb, lw=2.0, label="Subida")
    ax.plot(ws, tw_cruise, lw=2.0, label="Cruzeiro")
    ax.scatter(
        ws_project,
        tw_project,
        color="magenta",
        marker="x",
        s=70,
        zorder=6,
        label="Ponto de projeto",
    )
    w0 = INPUTS["w0"]
    s_project = w0 / ws_project
    t0_project = tw_project * w0
    ax.annotate(
        f"$W_0/S$ = {ws_project:.2f} N/m²\n"
        f"$T_0/W_0$ = {tw_project:.4f}\n"
        f"$S$ = {s_project:.2f} m²\n"
        f"$T_0$ = {t0_project/1000:.1f} kN",
        (ws_project, tw_project),
        textcoords="offset points",
        xytext=(12, 12),
        fontsize=9,
        clip_on=False,
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.9),
    )

    # Pouso entra como limite em W/S (linha vertical).
    ax.axvline(ws_landing_limit, color="red", ls="-", lw=2.4, zorder=5, label="Limite de pouso")

    ax.set_xlabel("W0/S [N/m^2]")
    ax.set_ylabel("T0/W0 [-]")
    ax.set_title("Diagrama de Projeto")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(ws_min_base, ws_max_base)
    ax.set_ylim(bottom=0.0)
    ax.legend()

    # Margens (folgas) do ponto de projeto em relacao a cada restricao.
    # Convencao: margem > 0 atende; margem < 0 nao atende.
    tw_takeoff_proj = float(takeoff_constraint(np.array([ws_project]), INPUTS)[0])
    tw_climb_proj = float(climb_constraint(np.array([ws_project]), INPUTS)[0])
    tw_cruise_proj = float(cruise_constraint(np.array([ws_project]), INPUTS)[0])

    margin_takeoff = tw_project - tw_takeoff_proj
    margin_climb = tw_project - tw_climb_proj
    margin_cruise = tw_project - tw_cruise_proj
    margin_landing = ws_landing_limit - ws_project
    t0_fixed = INPUTS["t0_fixed"]

    thrust_margin_takeoff = margin_takeoff * w0
    thrust_margin_climb = margin_climb * w0
    thrust_margin_cruise = margin_cruise * w0

    # Margens horizontais (W0/S) com tracao fixa.
    tw_fixed = t0_fixed / w0
    sigma = INPUTS["rho_to"] / 1.225
    coef_takeoff = 0.2387 / (sigma * INPUTS["clmax_to"] * INPUTS["d_to"])
    ws_takeoff_limit_fixed_t0 = tw_fixed / coef_takeoff
    margin_ws_takeoff_fixed_t0 = ws_takeoff_limit_fixed_t0 - ws_project
    margin_ws_landing = ws_landing_limit - ws_project
    s_takeoff_limit_fixed_t0 = w0 / ws_takeoff_limit_fixed_t0
    s_landing_limit = w0 / ws_landing_limit
    delta_s_takeoff = s_takeoff_limit_fixed_t0 - s_project
    delta_s_landing = s_landing_limit - s_project

    print("\n=== Margens do ponto de projeto ===")
    print(f"Ponto de projeto: W0/S = {ws_project:.2f} N/m^2, T0/W0 = {tw_project:.4f}")
    print(f"W0 adotado: {w0:.1f} N")
    print(f"Decolagem: T0/W0 requerido = {tw_takeoff_proj:.4f} | margem = {margin_takeoff:+.4f}")
    print(f"Subida:    T0/W0 requerido = {tw_climb_proj:.4f} | margem = {margin_climb:+.4f}")
    print(f"Cruzeiro:  T0/W0 requerido = {tw_cruise_proj:.4f} | margem = {margin_cruise:+.4f}")
    print(
        f"Pouso: limite W0/S = {ws_landing_limit:.2f} N/m^2 | "
        f"margem horizontal = {margin_landing:+.2f} N/m^2"
    )
    print("\n--- Margens de tracao (DeltaT = margem * W0) ---")
    print(f"Decolagem: DeltaT = {thrust_margin_takeoff:+.1f} N")
    print(f"Subida:    DeltaT = {thrust_margin_climb:+.1f} N")
    print(f"Cruzeiro:  DeltaT = {thrust_margin_cruise:+.1f} N")
    print("\n--- Margens horizontais (W0/S) com T0 fixo ---")
    print(f"T0 fixo adotado: {t0_fixed:.1f} N  ->  (T0/W0)fixo = {tw_fixed:.5f}")
    print(
        f"Decolagem: limite W0/S (T0 fixo) = {ws_takeoff_limit_fixed_t0:.2f} N/m^2 | "
        f"margem horizontal = {margin_ws_takeoff_fixed_t0:+.2f} N/m^2"
    )
    print(
        f"Pouso:     limite W0/S = {ws_landing_limit:.2f} N/m^2 | "
        f"margem horizontal = {margin_ws_landing:+.2f} N/m^2"
    )
    print("\n--- Conversao para area de asa (S = W0/(W0/S)) ---")
    print(f"S no ponto de projeto = {s_project:.2f} m^2")
    print(
        f"Decolagem (T0 fixo): S_limite = {s_takeoff_limit_fixed_t0:.2f} m^2 | "
        f"margem em area = {delta_s_takeoff:+.2f} m^2"
    )
    print(
        f"Pouso:              S_limite = {s_landing_limit:.2f} m^2 | "
        f"margem em area = {delta_s_landing:+.2f} m^2"
    )

    fig.tight_layout()
    fig.savefig("diagrama_projeto.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
