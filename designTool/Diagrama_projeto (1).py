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


# ============================================================================
# ENTRADAS (PREENCHER)
# ============================================================================
INPUTS = {
    # Faixa de W/S para varrer no grafico [N/m^2]
    "ws_min": 1500,
    "ws_max": 14200,
    "n_points": 400,
    # Ponto de projeto
    "ws_project": 7144.5,    # [N/m^2]
    "tw_project": 0.266,      # [-]
    "w0": 3235671.2,          # [N] peso de decolagem para converter margem em tracao
    "t0_fixed": 862000.0,     # [N] tracao fixa para margens horizontais (area de asa)
    # Decolagem
    "rho_to": 1.225,          # [kg/m^3] densidade no aeroporto de decolagem
    "clmax_to": 2.3293,        # [-] CLmax na configuracao de decolagem
    "d_to": 2900,            # [m] distancia de decolagem requerida
    # Subida (2o segmento / OEI, conforme seu criterio)
    "ks": 1.20,              # [-] fator velocidade de estol
    "gamma_cl": 0.024,       # [-] gradiente de subida
    "n_eng": 2,              # [-] numero total de motores
    "n_eng_failed": 1,       # [-] motores inoperantes
    "cd0_cl": 0.031446,          # [-] CD0 em subida
    "k_cl": 0.0363,            # [-] fator K (polar parabolica) em subida
    "clmax_cl": 2.3293,        # [-] CLmax usado na subida
    "wcl_w0": 0.975,          # [-] razao W_CL/W0 (mudanca de peso)
    "tcl_t0": 1,          # [-] razao T_CL/T0 (mudanca de tracao)
    # Cruzeiro
    "rho_cruise": 0.303,      # [kg/m^3] densidade no cruzeiro
    "v_cruise": 250.8,   # [m/s] Mach 0.85 a 40.000 ft (ISA)
    "cd0_cr": 0.012373,          # [-] CD0 no cruzeiro
    "k_cr": 0.0357,            # [-] fator K no cruzeiro
    "wcr_w0": 0.99**2*0.95*0.98,          # [-] razao W_CR/W0
    "tcr_t0": 0.165,         # [-] razao T_CR/T0
    # Pouso
    "rho_landing": 1.125,     # [kg/m^3] densidade no pouso
    "clmax_landing": 2.66,   # [-] CLmax na configuracao de pouso
    "d_landing": 2900,       # [m] distancia de pouso requerida
    "wld_w0": 0.7393,          # [-] razao W_LD/W0 (correcao para W0/S)
    # Parametros fisicos/regulatorios do pouso
    "g": 9.80665,            # [m/s^2] gravidade
    "a_g": 0.5,              # [-] desaceleracao media (normalizada por g)
    "f_ld": 5.0 / 3.0,       # [-] fator de seguranca FAR
    "h_ld": 15.3,            # [m] altura de obstaculo no pouso
}


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
    w0 = INPUTS["w0"]
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
    s_project = w0 / ws_project
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
