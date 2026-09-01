'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Problema 4 - Grupo NJ-0502

Etapa 2 - Otimização multiobjetivo (MOGA / NSGA-II do pymoo).

Minimiza simultaneamente W0 e Wf da NJ-0502, com as MESMAS nove variáveis
e as MESMAS dezessete restrições do Problema 3.

O script roda dois casos:

  base  - o problema do Problema 3 tal e qual, com o teto de envergadura
          da letra E da OACI (b_w < 64,9 m);
  letraF- o mesmo problema com o teto de envergadura relaxado para a
          letra F / ADG VI (b_w < 79,9 m) e a bitola para 15,9 m.
          Serve para mostrar QUEM colapsa a frente de Pareto: no caso
          base, a categoria de aeródromo trunca o compromisso entre
          W0 e Wf antes que ele apareça.

Uso:  python run_4_2_moga.py [base|letraF|todos]
'''

# IMPORTS
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# O MOGA amostra a caixa inteira, e nos cantos divergentes o designTool
# emite overflow/divide-by-zero. Esses pontos são tratados pela penalidade
# de `opt_multi`, então os avisos só poluiriam o log.
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(all='ignore')

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from opt_multi import (MultiObjModel, NJ0502BiObj, TOL_VIAVEL, frame_from_X,
                       gravity)
import opt_common as oc

# =========================================

# SETUP

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'resultados_otimizacao_multiobjetivo')

# Parâmetros do MOGA (pergunta 2 do roteiro).
POP_SIZE = 100        # indivíduos por geração
N_GEN = 400           # número de gerações
SEED = 42             # semente, para que a corrida seja reprodutível

# População inicial parcialmente semeada. O roteiro manda partir da
# configuração final de PRJ-22; num GA isso se traduz em injetar esse
# ponto (e uma nuvem em volta dele) na primeira geração, deixando o
# restante aleatório para preservar diversidade. Sem isso o NSGA-II gasta
# ~10 gerações só para achar o primeiro indivíduo viável.
N_SEMEADOS = 10
DISPERSAO_SEMENTE = 0.02   # fração da largura da caixa

# Operadores: SBX + mutação polinomial, os padrões do NSGA-II.
ETA_CROSSOVER = 15
PROB_CROSSOVER = 0.9
ETA_MUTATION = 20

# Casos rodados.
CASOS = {
    'base': {
        'rotulo': 'letra E (roteiro)',
        'bounds': None,
        'B_W_MAX': 64.9,
        'WHEEL_SPAN_MAX': 13.9,
    },
    'letraF': {
        'rotulo': 'letra F (teto relaxado)',
        'bounds': None,
        'B_W_MAX': 79.9,
        'WHEEL_SPAN_MAX': 15.9,
    },
}

# =========================================


def _aplica_categoria(caso):
    '''
    Ajusta os tetos de pátio no módulo do Problema 3.

    As restrições de envergadura e bitola são lambdas que leem
    `oc.B_W_MAX` e `oc.WHEEL_SPAN_MAX` por fechamento, então basta
    reescrever essas constantes antes de montar o modelo.
    '''
    oc.B_W_MAX = caso['B_W_MAX']
    oc.WHEEL_SPAN_MAX = caso['WHEEL_SPAN_MAX']


def populacao_inicial(model):
    '''
    Monta a população inicial: a configuração de PRJ-22, uma nuvem em
    volta dela e o restante amostrado uniformemente na caixa.
    '''
    rng = np.random.default_rng(SEED)
    n = len(model.x0)
    largura = model.xu - model.xl

    nuvem = (model.x0
             + DISPERSAO_SEMENTE*largura*rng.normal(size=(N_SEMEADOS - 1, n)))
    aleatorios = rng.uniform(model.xl, model.xu,
                             size=(POP_SIZE - N_SEMEADOS, n))

    X0 = np.vstack([model.x0.reshape(1, -1), nuvem, aleatorios])
    return np.clip(X0, model.xl, model.xu)


def roda_moga(nome, caso):
    '''
    Roda o NSGA-II para um caso e grava os CSV correspondentes.
    '''
    _aplica_categoria(caso)

    print('=' * 78)
    print('  MOGA (NSGA-II) - caso "%s": %s' % (nome, caso['rotulo']))
    print('  b_w < %.1f m | b_mlg < %.1f m'
          % (caso['B_W_MAX'], caso['WHEEL_SPAN_MAX']))
    print('  população = %d | gerações = %d | semente = %d'
          % (POP_SIZE, N_GEN, SEED))
    print('  população inicial: %d semeados (PRJ-22) + %d aleatórios'
          % (N_SEMEADOS, POP_SIZE - N_SEMEADOS))
    print('=' * 78)

    model = MultiObjModel()
    problem = NJ0502BiObj(model=model)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=populacao_inicial(model),
        crossover=SBX(eta=ETA_CROSSOVER, prob=PROB_CROSSOVER),
        mutation=PM(eta=ETA_MUTATION),
        eliminate_duplicates=True,
    )

    t0 = time.time()
    result = pymoo_minimize(problem,
                            algorithm,
                            get_termination('n_gen', N_GEN),
                            seed=SEED,
                            save_history=True,
                            verbose=False)
    elapsed = time.time() - t0

    n_eval = result.algorithm.evaluator.n_eval

    print('  tempo            : %.1f s' % elapsed)
    print('  avaliações totais: %d  (= %d x %d)'
          % (n_eval, POP_SIZE, N_GEN))
    print('  aval. designTool : %d (pontos distintos; cache ativo)'
          % model.n_designTool)
    print('  pontos inválidos : %d (%.1f%% - designTool não fechou, '
          'penalizados)'
          % (model.n_invalido,
             100.0*model.n_invalido/max(1, model.n_designTool)))

    if result.X is None:
        print('  *** nenhuma solução viável encontrada ***')
        return None

    X = np.atleast_2d(result.X)
    F = np.atleast_2d(result.F)
    print('  soluções na frente: %d' % len(X))

    # -----------------------------------------------------------
    # Frente de Pareto em unidades físicas

    rows = frame_from_X(model, X, F)
    df = pd.DataFrame(rows)
    df = df.sort_values('W0_kgf').reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'moga_frente_%s.csv' % nome),
              index=False)

    n_inviavel = int((~df['viavel']).sum())
    print('  soluções inviáveis na frente: %d' % n_inviavel)
    print('  W0 [kgf]: %.2f  ->  %.2f  (amplitude %.2f)'
          % (df['W0_kgf'].min(), df['W0_kgf'].max(),
             df['W0_kgf'].max() - df['W0_kgf'].min()))
    print('  Wf [kgf]: %.2f  ->  %.2f  (amplitude %.2f)'
          % (df['Wf_kgf'].min(), df['Wf_kgf'].max(),
             df['Wf_kgf'].max() - df['Wf_kgf'].min()))

    # -----------------------------------------------------------
    # Histórico por geração (convergência)

    hist = []
    for gen, entry in enumerate(result.history, start=1):
        opt = entry.opt
        if opt is None or len(opt) == 0:
            continue
        Fg = opt.get('F')
        Gg = opt.get('G')
        viavel = (Gg <= TOL_VIAVEL).all(axis=1) if Gg is not None else None
        hist.append({
            'geracao': gen,
            'n_eval': entry.evaluator.n_eval,
            'n_frente': len(opt),
            'f1_min': float(Fg[:, 0].min()),
            'f2_min': float(Fg[:, 1].min()),
            'W0_min_kgf': float(Fg[:, 0].min()*model.W0_ref/gravity),
            'Wf_min_kgf': float(Fg[:, 1].min()*model.Wf_ref/gravity),
            'n_viaveis': int(viavel.sum()) if viavel is not None else -1,
        })
    df_hist = pd.DataFrame(hist)
    df_hist.to_csv(os.path.join(RESULTS_DIR, 'moga_hist_%s.csv' % nome),
                   index=False)

    # -----------------------------------------------------------
    # Resumo da corrida

    pd.DataFrame([{
        'caso': nome,
        'rotulo': caso['rotulo'],
        'B_W_MAX': caso['B_W_MAX'],
        'WHEEL_SPAN_MAX': caso['WHEEL_SPAN_MAX'],
        'pop_size': POP_SIZE,
        'n_gen': N_GEN,
        'seed': SEED,
        'n_semeados': N_SEMEADOS,
        'n_eval': int(n_eval),
        'n_designTool': model.n_designTool,
        'n_invalido': model.n_invalido,
        'tempo_s': elapsed,
        'n_frente': len(df),
        'n_inviavel': n_inviavel,
        'W0_min_kgf': df['W0_kgf'].min(),
        'W0_max_kgf': df['W0_kgf'].max(),
        'Wf_min_kgf': df['Wf_kgf'].min(),
        'Wf_max_kgf': df['Wf_kgf'].max(),
        'W0_ref_kgf': model.W0_ref/gravity,
        'Wf_ref_kgf': model.Wf_ref/gravity,
    }]).to_csv(os.path.join(RESULTS_DIR, 'moga_corrida_%s.csv' % nome),
               index=False)

    print('  gravado: moga_frente_%s.csv, moga_hist_%s.csv, '
          'moga_corrida_%s.csv\n' % (nome, nome, nome))
    return df


# =========================================

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    alvo = sys.argv[1] if len(sys.argv) > 1 else 'todos'
    nomes = list(CASOS.keys()) if alvo == 'todos' else [alvo]

    # Guarda os tetos originais para restaurar ao fim.
    b_ori, w_ori = oc.B_W_MAX, oc.WHEEL_SPAN_MAX
    try:
        for nome in nomes:
            roda_moga(nome, CASOS[nome])
    finally:
        oc.B_W_MAX, oc.WHEEL_SPAN_MAX = b_ori, w_ori

    print('  Arquivos gravados em %s/' % RESULTS_DIR)
