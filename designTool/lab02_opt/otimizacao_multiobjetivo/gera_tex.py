'''
PRJ-23 Lab 02 - Problema 4. Tabelas LaTeX a partir dos CSV.
Uso: python gera_tex.py
'''

import os
import re
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_P3_DIR = os.path.join(os.path.dirname(_HERE), 'otimizacao_NJ0502')
if _P3_DIR not in sys.path:
    sys.path.insert(0, _P3_DIR)

from opt_common import CONSTRAINTS, DESIGN_VARS

# =========================================

_LAB = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_LAB, 'resultados_otimizacao_multiobjetivo')
TEX_DIR = os.path.join(_LAB, 'tex_otimizacao_multiobjetivo')

ROTULOS_SEL = ['A (mín.\\ $W_0$)', 'B (intermediária)', 'C (mín.\\ $W_f$)']


def esc(s):
    return str(s).replace('_', r'\_')


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    return (('%.' + str(nd) + 'f') % v).replace('-', r'$-$')


def write(name, content):
    path = os.path.join(TEX_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('  gravado: %s' % path)


# =========================================


def tab_parametros():
    '''
    Configuração do MOGA: responde à pergunta 2 do roteiro.
    '''
    c = pd.read_csv(os.path.join(RESULTS_DIR,
                                 'moga_corrida_base.csv')).iloc[0]
    itens = [
        ('Algoritmo', 'NSGA-II (\\code{pymoo} 0.6.1)'),
        ('Indivíduos por geração (\\code{pop\\_size})', '%d' % c['pop_size']),
        ('Número de gerações (\\code{n\\_gen})', '%d' % c['n_gen']),
        ('Avaliações da função objetivo', _mil(c['n_eval'])),
        ('População inicial semeada com PRJ-22',
         '%d de %d' % (c['n_semeados'], c['pop_size'])),
        ('Cruzamento', 'SBX, $\\eta = 15$, $p = 0{,}9$'),
        ('Mutação', 'polinomial, $\\eta = 20$'),
        ('Semente aleatória', '%d' % c['seed']),
        ('Variáveis de projeto', '%d' % len(DESIGN_VARS)),
        ('Restrições de desigualdade', '%d' % len(CONSTRAINTS)),
        ('Tempo de parede [s]', fmt(c['tempo_s'], 1)),
        ('Pontos na frente final', '%d' % c['n_frente']),
        ('Pontos inválidos penalizados', '%d' % c['n_invalido']),
    ]
    lines = [r'\begin{tabular}{lr}', r'\toprule',
             r'Parâmetro & Valor \\', r'\midrule']
    for label, valor in itens:
        lines.append('%s & %s \\\\' % (label, valor))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_convergencia():
    '''
    MOGA contra as âncoras do SLSQP: responde à pergunta 3.
    '''
    anc = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_ancoras.csv'),
                      index_col=0)
    fr = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_frente_base.csv'))
    ref = pd.read_csv(os.path.join(RESULTS_DIR, 'ref_frente_eps.csv'))
    ref = ref[ref['caso'] == 'base']

    W0_anc = anc.loc['min W0', 'W0_kgf']
    Wf_anc = anc.loc['min Wf', 'Wf_kgf']
    W0_moga = fr['W0_kgf'].min()
    Wf_moga = fr['Wf_kgf'].min()

    linhas = [
        (r'Menor $W_0$ --- âncora SLSQP (Seção~3)', fmt(W0_anc, 2), '--'),
        (r'Menor $W_0$ --- MOGA', fmt(W0_moga, 2),
         fmt(100.0*(W0_moga/W0_anc - 1.0), 4)),
        (r'Menor $W_f$ --- âncora SLSQP', fmt(Wf_anc, 2), '--'),
        (r'Menor $W_f$ --- MOGA', fmt(Wf_moga, 2),
         fmt(100.0*(Wf_moga/Wf_anc - 1.0), 4)),
        (r'Menor $W_0$ --- frente de referência', 
         fmt(ref['W0_kgf'].min(), 2),
         fmt(100.0*(ref['W0_kgf'].min()/W0_anc - 1.0), 4)),
    ]
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Grandeza & Valor [kgf] & Desvio da âncora [\%] \\',
             r'\midrule']
    for a, b, c in linhas:
        lines.append('%s & %s & %s \\\\' % (a, b, c))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_amplitude():
    '''
    Amplitude da frente em cada categoria de aeródromo.
    '''
    anc = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_ancoras_todas.csv'))
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Categoria & $b_w$ máx.\ [m] & $\Delta W_0$ [kgf] & '
             r'$\Delta W_f$ [kgf] & $\Delta W_0/W_0$ \\',
             r'\midrule']
    tetos = {'base': 64.9, 'letraF': 79.9}
    nomes = {'base': 'Letra E (roteiro)', 'letraF': 'Letra F (relaxado)'}
    for caso in ('base', 'letraF'):
        d = anc[anc['caso'] == caso].set_index('ancora')
        dW0 = abs(d.loc['min Wf', 'W0_kgf'] - d.loc['min W0', 'W0_kgf'])
        dWf = abs(d.loc['min W0', 'Wf_kgf'] - d.loc['min Wf', 'Wf_kgf'])
        rel = dW0/d.loc['min W0', 'W0_kgf']
        lines.append('%s & %s & %s & %s & %s \\\\' % (
            nomes[caso], fmt(tetos[caso], 1), fmt(dW0, 2), fmt(dWf, 2),
            ('%.1e' % rel).replace('-', r'$-$')))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def _tab_sel(arquivo):
    df = pd.read_csv(os.path.join(RESULTS_DIR, arquivo))
    campos = [
        (r'$W_0$ [kgf]', 'W0_kgf', 2),
        (r'$W_f$ [kgf]', 'Wf_kgf', 2),
        (r'$b_w$ [m]', 'b_w', 3),
        (r'$S_w$ [m$^2$]', 'S_w', 2),
        (r'$AR_w$', 'AR_w', 4),
        (r'$\Lambda_w$ [deg]', 'sweep_w', 2),
        (r'$x_{r,w}$ [m]', 'xr_w', 3),
        (r'$C_{vt}$', 'Cvt', 5),
        (r'$L_{c,h}$', 'Lc_h', 3),
        (r'$L_{b,v}$', 'Lb_v', 4),
        (r'$z_{lg}$ [m]', 'z_lg', 3),
    ]
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             'Grandeza & ' + ' & '.join(ROTULOS_SEL) + r' \\',
             r'\midrule']
    for label, col, nd in campos:
        vals = []
        for _, row in df.iterrows():
            v = row[col]
            if col == 'sweep_w':
                v = v*180.0/np.pi
            vals.append(fmt(v, nd))
        lines.append('%s & %s \\\\' % (label, ' & '.join(vals)))
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def tab_sel_base():
    return _tab_sel('sel_base.csv')


def tab_sel_letraF():
    return _tab_sel('sel_letraF.csv')




# =========================================
# RENDERIZAÇÃO DO TEXTO DA SEÇÃO
#
# O arquivo otimizacao_multiobjetivo.tex.in traz marcadores @@NOME@@ nos
# pontos em que o texto cita um número. Eles são substituídos aqui pelos
# valores dos CSV, de modo que a prosa nunca saia de sincronia com os
# resultados. O .tex final é gerado, não editado à mão.

TEMPLATE = os.path.join(_HERE, 'otimizacao_multiobjetivo.tex.in')
DESTINO = os.path.join(_LAB, 'otimizacao_multiobjetivo.tex')


def _mil(v):
    '''Inteiro com separador de milhar do LaTeX (40\\,000).'''
    return '{:,}'.format(int(round(v))).replace(',', r'\,')


def _dec(v, nd=2):
    '''Decimal com vírgula, e separador de milhar quando couber.'''
    txt = ('%.' + str(nd) + 'f') % v
    inteira, _, frac = txt.partition('.')
    neg = inteira.startswith('-')
    inteira = inteira.lstrip('-')
    if len(inteira) > 4:
        inteira = '{:,}'.format(int(inteira)).replace(',', r'\,')
    saida = inteira + (('{,}' + frac) if frac else '')
    return ('$-$' if neg else '') + saida


def valores():
    '''Monta o dicionário de substituições a partir dos CSV.'''
    c = pd.read_csv(os.path.join(RESULTS_DIR,
                                 'moga_corrida_base.csv')).iloc[0]
    anc = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_ancoras_todas.csv'))
    fr = pd.read_csv(os.path.join(RESULTS_DIR, 'moga_frente_base.csv'))
    selE = pd.read_csv(os.path.join(RESULTS_DIR, 'sel_base.csv'))
    selF = pd.read_csv(os.path.join(RESULTS_DIR, 'sel_letraF.csv'))

    aE = anc[anc['caso'] == 'base'].set_index('ancora')
    aF = anc[anc['caso'] == 'letraF'].set_index('ancora')

    amp_W0_E = abs(aE.loc['min Wf', 'W0_kgf'] - aE.loc['min W0', 'W0_kgf'])
    amp_Wf_E = abs(aE.loc['min W0', 'Wf_kgf'] - aE.loc['min Wf', 'Wf_kgf'])
    amp_W0_F = abs(aF.loc['min Wf', 'W0_kgf'] - aF.loc['min W0', 'W0_kgf'])
    amp_Wf_F = abs(aF.loc['min W0', 'Wf_kgf'] - aF.loc['min Wf', 'Wf_kgf'])

    W0_anc = aE.loc['min W0', 'W0_kgf']
    Wf_anc = aE.loc['min Wf', 'Wf_kgf']
    W0_moga = fr['W0_kgf'].min()
    Wf_moga = fr['Wf_kgf'].min()

    # selE/selF vêm ordenadas por W0: A = menor W0, C = menor Wf.
    A_E, C_E = selE.iloc[0], selE.iloc[-1]
    A_F, C_F = selF.iloc[0], selF.iloc[-1]
    rad2deg = 180.0/np.pi

    return {
        'POP': _mil(c['pop_size']),
        'NGEN': _mil(c['n_gen']),
        'N_EVAL': _mil(c['n_eval']),
        'N_RAND': _mil(c['pop_size'] - c['n_semeados']),
        'TEMPO': _dec(c['tempo_s'], 0),
        'N_INVALIDO': _mil(c['n_invalido']),
        'N_FRENTE': _mil(c['n_frente']),

        'AMP_W0_E': _dec(amp_W0_E, 2),
        'AMP_WF_E': _dec(amp_Wf_E, 2),
        'AMP_W0_F': _dec(amp_W0_F, 2),
        'AMP_WF_F': _dec(amp_Wf_F, 2),
        'AMP_REL_E': _dec(100.0*amp_W0_E/W0_anc, 3) + r'\%',
        'RAZAO_AMP': _dec(amp_W0_F/amp_W0_E, 0),

        'GAP_W0': _dec(100.0*(W0_moga/W0_anc - 1.0), 2),
        'GAP_WF': _dec(100.0*(Wf_moga/Wf_anc - 1.0), 2),
        'GAP_W0_KGF': _mil(W0_moga - W0_anc),

        'BW_A_E': _dec(A_E['b_w'], 2),
        'BW_C_E': _dec(C_E['b_w'], 2),
        'DBW_E': _dec(abs(C_E['b_w'] - A_E['b_w']), 2),
        'DAR_E': _dec(abs(C_E['AR_w'] - A_E['AR_w']), 3),

        'BW_A_F': _dec(A_F['b_w'], 2),
        'BW_C_F': _dec(C_F['b_w'], 2),
        'AR_A_F': _dec(A_F['AR_w'], 2),
        'AR_C_F': _dec(C_F['AR_w'], 2),
        'SW_A_F': _dec(A_F['sweep_w']*rad2deg, 1),
        'SW_C_F': _dec(C_F['sweep_w']*rad2deg, 1),
        'S_A_F': _dec(A_F['S_w'], 1),
        'S_C_F': _dec(C_F['S_w'], 1),
    }


def renderiza():
    with open(TEMPLATE, encoding='utf-8') as f:
        texto = f.read()

    subs = valores()
    for chave, valor in subs.items():
        texto = texto.replace('@@%s@@' % chave, valor)

    faltando = sorted(set(re.findall(r'@@([A-Z_0-9]+)@@', texto)))
    if faltando:
        raise RuntimeError('marcadores sem valor: %s' % ', '.join(faltando))

    with open(DESTINO, 'w', encoding='utf-8') as f:
        f.write(texto)
    print('  gravado: %s  (%d marcadores substituídos)'
          % (DESTINO, len(subs)))


# =========================================

if __name__ == '__main__':

    os.makedirs(TEX_DIR, exist_ok=True)

    write('tab_parametros.tex', tab_parametros())
    write('tab_convergencia.tex', tab_convergencia())
    write('tab_amplitude.tex', tab_amplitude())
    write('tab_sel_base.tex', tab_sel_base())
    write('tab_sel_letraF.tex', tab_sel_letraF())

    renderiza()
