'''
INSTITUTO TECNOLOGICO DE AERONAUTICA
PRJ-23 - Aircraft Preliminary Design
Homework 01 - DOE analysis - Grupo NJ-0502

Modulo auxiliar comum aos tres scripts do laboratorio.

Concentra:
  - copia profunda e escrita de inputs (inclusive aninhados, ex.: 'engine.BPR');
  - execucao protegida da funcao `analyze` do designTool (timeout + guarda de
    divergencia dos lacos de ponto fixo de `weight`/`thrust_matching`);
  - extracao padronizada das saidas de interesse do dicionario `airplane`.

Maj. Ney Secco fornece o designTool; este arquivo apenas o instrumenta.
'''

# IMPORTS
import copy
import signal

import numpy as np

from designTool.analyze import analyze
from designTool.standard_airplane import standard_airplane
from designTool.constants import gravity

# =========================================
# CONSTANTES

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0

# Tempo maximo (s) permitido para uma unica execucao de `analyze`.
# Os lacos `while abs(delta) > 10` de weight() e thrust_matching() nao possuem
# limite de iteracoes: para combinacoes extremas de entrada o ponto fixo pode
# divergir e nunca retornar. O alarme abaixo transforma esse caso em falha
# controlada da amostra em vez de travar o DOE.
TIMEOUT_S = 5.0


class _AnalyzeTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _AnalyzeTimeout()


# =========================================
# MANIPULACAO DE INPUTS


def get_baseline(name):
    '''
    Retorna o dicionario de inputs da aeronave de referencia.
    name: 'fokker100' (aeronave padrao) ou 'my_airplane' (aeronave do grupo).
    '''
    return standard_airplane(name)


def get_input(airplane, key):
    '''
    Le um input. Aceita chaves aninhadas no formato 'engine.BPR'.
    '''
    node = airplane['inputs']
    parts = key.split('.')
    for p in parts[:-1]:
        node = node[p]
    return node[parts[-1]]


def set_input(airplane, key, value):
    '''
    Escreve um input. Aceita chaves aninhadas no formato 'engine.BPR'.
    '''
    node = airplane['inputs']
    parts = key.split('.')
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = value


def perturb(baseline_name, key, value):
    '''
    Devolve uma copia da aeronave de referencia com um unico input alterado.
    A copia profunda e obrigatoria porque `analyze` escreve no proprio
    dicionario (chaves 'geometry', 'thrust_matching', 'balance', ...).
    '''
    airplane = copy.deepcopy(get_baseline(baseline_name))
    set_input(airplane, key, value)
    return airplane


# =========================================
# EXECUCAO PROTEGIDA


# Saidas extraidas de cada execucao. A chave e o nome usado nas tabelas e
# graficos; o valor e (caminho no dicionario, fator de conversao).
OUTPUT_MAP = {
    'W0':               (('thrust_matching', 'W0'),               1.0 / gravity),  # kgf
    'W_f':              (('thrust_matching', 'W_fuel'),           1.0 / gravity),  # kgf
    'W_e':              (('thrust_matching', 'W_empty'),          1.0 / gravity),  # kgf
    'T0':               (('thrust_matching', 'T0'),               1.0 / gravity),  # kgf
    'deltaS_wlan':      (('thrust_matching', 'deltaS_wlan'),      1.0),            # m2
    'SM_fwd':           (('balance', 'SM_fwd'),                   1.0),            # -
    'SM_aft':           (('balance', 'SM_aft'),                   1.0),            # -
    'CLv':              (('balance', 'CLv'),                      1.0),            # -
    'V_maxfuel':        (('balance', 'V_maxfuel'),                1000.0),         # L
    'tank_excess':      (('balance', 'tank_excess'),              1.0),            # -
    'frac_nlg_fwd':     (('landing_gear', 'frac_nlg_fwd'),        1.0),            # -
    'frac_nlg_aft':     (('landing_gear', 'frac_nlg_aft'),        1.0),            # -
    'alpha_tipback':    (('landing_gear', 'alpha_tipback'),       rad2deg),        # deg
    'alpha_tailstrike': (('landing_gear', 'alpha_tailstrike'),    rad2deg),        # deg
    'phi_overturn':     (('landing_gear', 'phi_overturn'),        rad2deg),        # deg
}


def extract_outputs(airplane):
    '''
    Le as saidas de interesse do dicionario ja processado por `analyze`.
    '''
    out = {}
    for name, (path, factor) in OUTPUT_MAP.items():
        value = airplane
        for p in path:
            value = value[p]
        out[name] = np.nan if value is None else float(value) * factor
    return out


def run_case(airplane, timeout=TIMEOUT_S):
    '''
    Executa `analyze` de forma protegida.

    Retorna o dicionario de saidas, ou None se a analise falhar (excecao
    numerica, divergencia do ponto fixo detectada pelo timeout, ou saida
    nao-finita). Isso e necessario para o DOE: as amostras do LHS varrem
    faixas amplas e algumas combinacoes nao produzem uma aeronave viavel.
    '''
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with np.errstate(all='ignore'):
            analyze(airplane, print_log=False, plot=False)
        out = extract_outputs(airplane)
    except (_AnalyzeTimeout, Exception):
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)

    # Descarta resultados nao-finitos ou fisicamente absurdos
    for name, value in out.items():
        if not np.isfinite(value):
            return None
    if out['W0'] <= 0 or out['W_f'] <= 0 or out['W_e'] <= 0:
        return None

    return out


def run_baseline(baseline_name):
    '''
    Executa a aeronave de referencia sem nenhuma perturbacao.
    '''
    return run_case(copy.deepcopy(get_baseline(baseline_name)))
