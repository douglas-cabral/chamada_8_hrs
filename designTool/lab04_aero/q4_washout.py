'''
PRJ-23 Lab 04 - Estudo de torcao geometrica (washout) da asa.

O item 4 mostrou que, com torcao nula, a secao critica cai em eta = 0,91 --
dentro da faixa do aileron -- e que o C_Lmax de asa limpa fica bem abaixo do
que o designTool supoe. Este script quantifica o remedio classico: torcao
linear de 0 graus na raiz a TWIST graus na ponta, varrida em varios valores.

Para cada washout o estudo refaz a cadeia inteira, e nao so o item 4:

  1. reajusta a incidencia de empenagem i_t (item 3), porque a torcao muda o
     angulo de sustentacao nula e o momento da asa; manter o i_t antigo
     misturaria o efeito do washout com um erro de trimagem;
  2. le, no ponto de projeto de cruzeiro, o C_D, o fator de Oswald e a
     eficiencia L/D com esse i_t (insumo do item 5);
  3. refaz o metodo da secao critica em baixa velocidade, nos quatro casos do
     item 4, obtendo alpha_max, C_Lmax, delta_e e a estacao do estol.

O criterio de escolha e o requisito de decolagem do proprio designTool
(performance.py): a pista de s_TO exige

    C_Lmax,TO >= 0,2387 (W_0/S_w) / (sigma (T_0/W_0) s_TO),

e o C_Lmax de decolagem e estimado como o C_Lmax de asa limpa medido aqui mais
o incremento de flap e slat do Raymer que o designTool ja calcula. A regra e:
escolhe-se a menor torcao que atenda o requisito com a margem MARGEM; se
nenhuma atender, a menor que atenda o requisito sem margem; se ainda assim
nenhuma atender, a maior torcao varrida. O que valeu fica registrado no JSON.

Custo: o alpha de estol sai de um modelo linear montado com duas corridas
(alpha = 0 e 12 graus) e e depois corrigido por Newton sobre o resultado real
do AVL, porque o AVL nao e exatamente linear em alpha -- a direcao da corrente
livre gira com o angulo de ataque. O desvio entre o modelo linear e o valor
convergido e reportado.

Uso (a partir desta pasta):
    python q4_washout.py           # roda o AVL e grava tudo  (~30 min)
    python q4_washout.py --tex     # refaz so CSV/tabelas/figuras
Saidas: resultados_washout/ e tex_washout/
'''

import csv
import json
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import matplotlib.pyplot as plt

LAB = os.path.dirname(os.path.abspath(__file__))
for p in (LAB, os.path.abspath(os.path.join(LAB, '..')),
          os.path.abspath(os.path.join(LAB, '..', 'lab02_opt',
                                       'otimizacao_NJ0502'))):
    if p not in sys.path:
        sys.path.insert(0, p)

import avl_tools as at                     # noqa: E402
import ponto_projeto as pj                 # noqa: E402
import xfoil_perfil as xp                  # noqa: E402
import opt_common as oc                    # noqa: E402
from designTool.aerodynamics import aerodynamics    # noqa: E402
from designTool.auxiliary import atmosphere         # noqa: E402

OUT = os.path.join(LAB, 'resultados_washout')
TEX = os.path.join(LAB, 'tex_washout')

WASHOUTS = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
MARGEM = 0.05                  # margem exigida sobre o requisito de decolagem
CASOS = [('fwd_livre', 'fwd', False, 'CG dianteiro, sem trimagem'),
         ('fwd_trim',  'fwd', True,  'CG dianteiro, com trimagem'),
         ('aft_livre', 'aft', False, 'CG traseiro, sem trimagem'),
         ('aft_trim',  'aft', True,  'CG traseiro, com trimagem')]
CASO_CRITICO = 'fwd_trim'      # o que dimensiona, conforme o item 4
CASO_AJUSTE = 'aft_livre'      # item 5.b do enunciado

IT_SONDA = (-3.0, -1.0)        # dois i_t de sondagem para a secante
A_SONDA = (0.0, 12.0)          # dois alphas de sondagem do modelo linear
A_GRADE = np.arange(0.0, 26.0, 0.02)
N_ITER = 4                     # iteracoes do Newton sobre o alpha de estol
TOL_RAZAO = 2e-4               # tolerancia em max(cl/cl_max) = 1


# ------------------------------------------------------------- geracao do avl

def escreve_avl(origem, destino, twist_tip, b2):
    '''Copia um .avl aplicando torcao linear de 0 na raiz a twist_tip na ponta.

    So mexe no campo Ainc das secoes da superficie Wing; empenagens, nacele,
    fuselagem, Xref e CDp ficam intactos.
    '''
    linhas = open(os.path.join(at.AVLDIR, origem),
                  encoding='utf-8').read().splitlines()
    saida, surf, k = [], None, 0
    while k < len(linhas):
        ln = linhas[k]
        saida.append(ln)
        if ln.strip().upper() == 'SURFACE':
            surf = linhas[k + 1].strip()
        if ln.strip().upper() == 'SECTION' and surf == 'Wing':
            j = k + 1
            while not linhas[j].strip():
                saida.append(linhas[j])
                j += 1
            campos = linhas[j].split()
            campos[4] = '%.4f' % (twist_tip*float(campos[1])/b2)
            saida.append(' '.join(campos))
            k = j + 1
            continue
        k += 1
    caminho = os.path.join(at.AVLDIR, destino)
    with open(caminho, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(saida) + '\n')
    return destino


# --------------------------------------------------------------- nucleo AVL

def resolve_it(arq, mach, CL, tag):
    '''i_t que anula o profundor no ponto de projeto (item 3, por secante).

    Como o AVL e linear, dois pontos bastam: delta_e(i_t) e uma reta.
    '''
    des = []
    for n, it in enumerate(IT_SONDA):
        r = at.run_cases(arq, mach, [{'CL': CL, 'trim': True}], it=it,
                         tag='%s_it%d' % (tag, n))[0]
        des.append(r['elevator'])
    i0, i1 = IT_SONDA
    return i0 + (0.0 - des[0])*(i1 - i0)/(des[1] - des[0])


def modelo_linear(p0, p1, lim):
    '''cl_norm(y, alpha)/cl_max a partir de duas corridas do AVL.'''
    w0, w1 = at.wing_strips(p0['fs']), at.wing_strips(p1['fs'])
    a0, a1 = A_SONDA
    A = w0['cl_norm']/lim
    B = (w1['cl_norm'] - w0['cl_norm'])/lim/(a1 - a0)
    raz = (A[:, None] + B[:, None]*(A_GRADE - a0)[None, :]).max(axis=0)
    CL = p0['CLtot'] + (p1['CLtot'] - p0['CLtot'])*(A_GRADE - a0)/(a1 - a0)
    return raz, CL


def torcao_para(ws, cl, alvo):
    '''Torcao que entrega um dado C_Lmax de asa limpa.

    Dentro da faixa varrida usa interpolacao; fora dela, a reta de minimos
    quadrados ajustada a C_Lmax(eps_t), que e quase exatamente linear.
    '''
    ws, cl = np.asarray(ws, dtype=float), np.asarray(cl, dtype=float)
    o = np.argsort(cl)
    if cl.min() <= alvo <= cl.max():
        return float(np.interp(alvo, cl[o], ws[o]))
    a, b = np.polyfit(ws, cl, 1)
    return float((alvo - b)/a)


def alpha_do_estol(raz):
    '''Alpha em que max_y(cl/cl_max) cruza a unidade.'''
    if raz[0] >= 1.0:
        raise RuntimeError('ja estolado em alpha=%.2f' % A_GRADE[0])
    if raz[-1] <= 1.0:
        raise RuntimeError('sem estol ate alpha=%.2f (razao=%.4f)'
                           % (A_GRADE[-1], raz[-1]))
    k = int(np.argmax(raz >= 1.0))
    return float(np.interp(1.0, raz[k-1:k+1], A_GRADE[k-1:k+1]))


def analisa(arq, cg, mach_cr, CL_proj, mach_bv, clmax_fn, Re_m, b2, tag):
    '''Cadeia completa de um arquivo .avl: i_t, cruzeiro e estol.'''
    it = resolve_it(arq, mach_cr, CL_proj, tag)
    cr = at.run_cases(arq, mach_cr, [{'CL': CL_proj, 'trim': True}], it=it,
                      tag=tag + '_cr')[0]

    sonda = at.run_cases(
        arq, mach_bv,
        [{'alpha': A_SONDA[0]}, {'alpha': A_SONDA[1]},
         {'alpha': A_SONDA[0], 'trim': True},
         {'alpha': A_SONDA[1], 'trim': True}],
        it=it, tag=tag + '_sd')
    lim = np.asarray(clmax_fn(Re_m*at.wing_strips(sonda[0]['fs'])['Chord']),
                     dtype=float)

    prev, incl = {}, {}
    for nome, (i0, i1) in (('livre', (0, 1)), ('trim', (2, 3))):
        raz, CL = modelo_linear(sonda[i0], sonda[i1], lim)
        a = alpha_do_estol(raz)
        prev[nome] = (a, float(np.interp(a, A_GRADE, CL)))
        incl[nome] = float(np.interp(a + 0.25, A_GRADE, raz)
                           - np.interp(a - 0.25, A_GRADE, raz))/0.5

    # o AVL nao e exatamente linear em alpha (a direcao da corrente livre gira
    # com o angulo de ataque), entao o alpha previsto pelo modelo e so o chute
    # inicial de um Newton sobre o resultado real do AVL
    alpha = {n: prev[n][0] for n in prev}
    for k in range(N_ITER):
        conf = at.run_cases(arq, mach_bv,
                            [{'alpha': alpha['livre']},
                             {'alpha': alpha['trim'], 'trim': True}],
                            it=it, tag='%s_cf%d' % (tag, k))
        raz_real = {}
        for nome, r in zip(('livre', 'trim'), conf):
            w = at.wing_strips(r['fs'])
            raz_real[nome] = float(np.nanmax(w['cl_norm']/lim))
        if max(abs(v - 1.0) for v in raz_real.values()) < TOL_RAZAO:
            break
        for nome, v in raz_real.items():
            alpha[nome] += (1.0 - v)/incl[nome]

    saida = {'it_deg': float(it), 'arquivo': arq,
             'cruzeiro': {'CD': float(cr['CDtot']), 'e': float(cr['e']),
                          'CDind': float(cr['CDind']),
                          'alpha_deg': float(cr['Alpha']),
                          'delta_e_deg': float(cr['elevator']),
                          'CL': float(cr['CLtot'])}}
    for (nome, (_, CL_prev)), r in zip(prev.items(), conf):
        a_prev = alpha[nome]
        w = at.wing_strips(r['fs'])
        raz = w['cl_norm']/lim
        i = int(np.argmax(raz))
        plat = (w['Yle']/b2)[raz > 0.97]
        saida[cg + '_' + nome] = {
            'alpha_max_deg': float(a_prev), 'CLmax': float(r['CLtot']),
            'CLmax_previsto': float(CL_prev),
            'erro_linear': float(r['CLtot'] - CL_prev),
            'delta_e_deg': float(r['elevator']),
            'eta_estol': float(w['Yle'][i]/b2),
            'razao_final': float(raz[i]),
            'eta_plat_ini': float(plat.min()), 'eta_plat_fim': float(plat.max()),
            'dist': {'eta': (w['Yle']/b2).tolist(),
                     'cl_norm': w['cl_norm'].tolist(),
                     'clmax_lim': lim.tolist(), 'razao': raz.tolist()}}
    return saida


def polares(w, pp, caso, b2, n_pontos=28, CL_min=-0.5):
    '''Polares completas do item 5 para um washout, no ponto de projeto.

    As polares de torcao nula sao lidas de resultados_q5/, gravadas pelo
    q5_polares.py, para que a comparacao seja com o resultado oficial do
    item 5 e nao com uma segunda corrida.
    '''
    saida = {'washout_deg': float(w), 'casos': {}}
    try:
        for cg in ('fwd', 'aft'):
            escreve_avl('%s.avl' % cg, 'wash_%s.avl' % cg, w, b2)
        for nome, cg, trim, _ in CASOS:
            CLmax = caso[nome]['CLmax']
            CLs = np.unique(np.concatenate(
                [np.linspace(CL_min, CLmax, n_pontos), [pp['CL']]]))
            rs = at.run_cases('wash_%s.avl' % cg, pp['M'],
                              [{'CL': float(c), 'trim': trim} for c in CLs],
                              it=caso['it_%s' % cg], tag='pol_' + nome)
            d = {k: [float(r[v]) for r in rs] for k, v in
                 (('CL', 'CLtot'), ('CD', 'CDtot'), ('CDind', 'CDind'),
                  ('alpha', 'Alpha'), ('delta_e', 'elevator'), ('e', 'e'))}
            i = int(np.argmin(np.abs(np.array(d['CL']) - pp['CL'])))
            d.update({'i_projeto': i, 'CLmax': float(CLmax),
                      'CD_projeto': d['CD'][i],
                      'alpha_projeto': d['alpha'][i],
                      'delta_e_projeto': d['delta_e'][i]})
            saida['casos'][nome] = d
            print('  polar %-10s CLmax=%.4f  CD(ponto)=%.5f  alpha=%+.3f  '
                  'delta_e=%+.3f' % (nome, CLmax, d['CD_projeto'],
                                     d['alpha_projeto'], d['delta_e_projeto']))
    finally:
        for cg in ('fwd', 'aft'):
            f = os.path.join(at.AVLDIR, 'wash_%s.avl' % cg)
            if os.path.exists(f):
                os.remove(f)

    # ajuste CD = CD0 + CDa*alpha + CDa2*alpha^2 (Tab. 7 do enunciado)
    d = saida['casos'][CASO_AJUSTE]
    CL, CD, al = (np.array(d['CL']), np.array(d['CD']), np.array(d['alpha']))
    m = (CL >= -0.1) & (CL <= d['CLmax'])
    coef = np.polyfit(np.radians(al[m]), CD[m], 2)
    saida['ajuste'] = {
        'caso': CASO_AJUSTE, 'CD0': float(coef[2]),
        'CD_alpha_por_rad': float(coef[1]),
        'CD_alpha2_por_rad2': float(coef[0]),
        'r2': float(1 - np.sum((CD[m] - np.polyval(coef, np.radians(al[m])))**2)
                    / np.sum((CD[m] - CD[m].mean())**2))}
    print('  ajuste CD(alpha) com washout: CD0=%.5f  CDa=%.5f  CDa2=%.5f  '
          'R2=%.5f' % (saida['ajuste']['CD0'],
                       saida['ajuste']['CD_alpha_por_rad'],
                       saida['ajuste']['CD_alpha2_por_rad2'],
                       saida['ajuste']['r2']))

    ref = os.path.join(LAB, 'resultados_q5', 'q5_polares.json')
    if os.path.exists(ref):
        with open(ref, encoding='utf-8') as f:
            q5 = json.load(f)
        saida['referencia_q5'] = {
            'polares': {k: {kk: q5['polares'][k][kk] for kk in
                            ('CL', 'CD', 'alpha', 'delta_e', 'CLmax',
                             'CD_projeto', 'alpha_projeto',
                             'delta_e_projeto')}
                        for k in q5['polares']},
            'ajuste': q5['ajuste_quadratico'],
            'CD_designTool': q5['CD_designTool_no_ponto'],
            'polar_designTool': q5['polar_designTool']}
    return saida


# ------------------------------------------------------------------ calculo

def requisitos(ap):
    '''C_Lmax exigido em decolagem e pouso, pelas formulas do designTool.'''
    I, T = ap['inputs'], ap['thrust_matching']
    W0, S_w = T['W0'], I['S_w']
    sigma = atmosphere(I['altitude_takeoff'],
                       I['deltaISA_takeoff'])['density']/1.225
    # performance.py: T0/W0 = 0.2387/sigma/CLmaxTO/s_TO*(W0/S_w)
    req_TO = (0.2387/sigma/I['distance_takeoff']*(W0/S_w)
              / (T['T0']/W0))

    clmax = {}
    for cfg in ('clean', 'takeoff', 'landing'):
        _, cl, _ = aerodynamics(ap, 0.2, I['altitude_takeoff'], 0.5,
                                n_engines_failed=0, highlift_config=cfg,
                                lg_down=1, h_ground=I['h_ground'])
        clmax[cfg] = float(cl)

    # S_wlan ~ 1/CLmaxLD, entao o CLmaxLD que faz S_wlan = S_w e proporcional
    S_wlan = S_w - T['deltaS_wlan']
    req_LD = clmax['landing']*S_wlan/S_w

    return {'CLmaxTO_req': float(req_TO), 'CLmaxLD_req': float(req_LD),
            'CLmax_clean_DT': clmax['clean'],
            'CLmaxTO_DT': clmax['takeoff'], 'CLmaxLD_DT': clmax['landing'],
            'dCL_TO': clmax['takeoff'] - clmax['clean'],
            'dCL_LD': clmax['landing'] - clmax['clean'],
            'clean_nec_TO': float(req_TO - (clmax['takeoff'] - clmax['clean'])),
            'clean_nec_LD': float(req_LD - (clmax['landing'] - clmax['clean'])),
            'clean_alvo': float((1 + MARGEM)*req_TO
                                - (clmax['takeoff'] - clmax['clean'])),
            'margem': MARGEM,
            'W0_S': float(W0/S_w), 'T0_W0': float(T['T0']/W0),
            's_TO': float(I['distance_takeoff']),
            's_LD': float(I['distance_landing'])}


def calcula():
    ap = oc.run_designTool(oc.get_baseline())
    I, G = ap['inputs'], ap['geometry']
    pp = pj.ponto_projeto(ap)
    bv = pj.condicao_baixa_velocidade(ap)
    b2 = 0.5*G['b_w']
    Re_m = bv['Re_por_metro']
    _, clmax_fn = xp.carrega(mach=bv['M'], plano='corrente')
    req = requisitos(ap)

    print('Requisitos do designTool:')
    print('  decolagem: C_LmaxTO >= %.4f  (W0/S=%.0f N/m2, T0/W0=%.4f, '
          's=%.0f m)' % (req['CLmaxTO_req'], req['W0_S'], req['T0_W0'],
                         req['s_TO']))
    print('  pouso    : C_LmaxLD >= %.4f  (Torenbeek, s=%.0f m)'
          % (req['CLmaxLD_req'], req['s_LD']))
    print('  incrementos de alta sustentacao (Raymer, designTool): '
          'flap+slat TO %+.4f, pouso %+.4f' % (req['dCL_TO'], req['dCL_LD']))
    print('  logo, asa limpa precisa de %.4f (decolagem) e %.4f (pouso);'
          % (req['clean_nec_TO'], req['clean_nec_LD']))
    print('  com %.0f%% de margem na decolagem: %.4f\n'
          % (100*MARGEM, req['clean_alvo']))
    print('Varredura de washout (M_cruzeiro=%.2f, M_baixa=%.4f, '
          'V2=%.1f m/s, i_w=%+.1f deg)'
          % (pp['M'], bv['M'], bv['V2'], pp['iw_deg']))
    print('%7s %8s %8s %9s %8s %8s %9s %7s'
          % ('washout', 'it_fwd', 'it_aft', 'CLmax*', 'alpha*', 'eta*',
             'CD_cruz', 'e'))

    casos = {}
    try:
        for w in WASHOUTS:
            tw = 'w%02d' % abs(int(round(10*w)))
            d = {'washout_deg': float(w)}
            for cg in ('fwd', 'aft'):
                arq = escreve_avl('%s.avl' % cg, 'wash_%s.avl' % cg, w, b2)
                r = analisa(arq, cg, pp['M'], pp['CL'], bv['M'], clmax_fn,
                            Re_m, b2, '%s_%s' % (tw, cg))
                d['it_%s' % cg] = r.pop('it_deg')
                d['cruz_%s' % cg] = r.pop('cruzeiro')
                r.pop('arquivo')
                d.update(r)
            c = d[CASO_CRITICO]
            print('%+6.1f  %+8.3f %+8.3f %9.4f %8.2f %8.3f %9.5f %7.4f'
                  % (w, d['it_fwd'], d['it_aft'], c['CLmax'],
                     c['alpha_max_deg'], c['eta_estol'],
                     d['cruz_fwd']['CD'], d['cruz_fwd']['e']))
            casos[str(w)] = d
    finally:
        for cg in ('fwd', 'aft'):
            f = os.path.join(at.AVLDIR, 'wash_%s.avl' % cg)
            if os.path.exists(f):
                os.remove(f)

    erro = max(abs(casos[k][n]['erro_linear'])
               for k in casos for n, _, _, _ in CASOS)
    folga = max(abs(casos[k][n]['razao_final'] - 1.0)
                for k in casos for n, _, _, _ in CASOS)
    print('\nconvergencia do alpha de estol: |max(cl/cl_max) - 1| <= %.1e'
          % folga)
    print('desvio do modelo linear em C_Lmax (nao linearidade do AVL): %.2e'
          % erro)

    # ------------------------------------------------ escolha do washout
    ws = np.array(WASHOUTS)
    cl = np.array([casos[str(w)][CASO_CRITICO]['CLmax'] for w in WASHOUTS])
    o = np.argsort(-ws)          # do menos torcido ao mais torcido: cl cresce

    atende_alvo = [w for w in WASHOUTS
                   if casos[str(w)][CASO_CRITICO]['CLmax'] >= req['clean_alvo']]
    atende_req = [w for w in WASHOUTS
                  if casos[str(w)][CASO_CRITICO]['CLmax'] >= req['clean_nec_TO']]
    if atende_alvo:
        escolha, regra = max(atende_alvo), 'requisito de decolagem com margem'
    elif atende_req:
        escolha, regra = max(atende_req), 'requisito de decolagem sem margem'
    else:
        escolha, regra = min(WASHOUTS), 'maior torcao varrida (requisito nao atendido)'

    print('washout escolhido: %+.1f deg  (%s)' % (escolha, regra))
    print('  torcao continua que zera a folga: %.2f deg; com %.0f%% de margem: '
          '%.2f deg' % (torcao_para(ws, cl, req['clean_nec_TO']), 100*MARGEM,
                        torcao_para(ws, cl, req['clean_alvo'])))

    print('\nPolares do item 5 com o washout escolhido:')
    pol = polares(escolha, pp, casos[str(escolha)], b2)

    return {'ponto_projeto': pp, 'baixa_velocidade': bv, 'requisitos': req,
            'washouts': WASHOUTS, 'casos': casos,
            'caso_critico': CASO_CRITICO,
            'escolha_deg': float(escolha), 'regra_escolha': regra,
            'erro_linearidade': float(erro),
            'folga_convergencia': float(folga),
            'polares': pol,
            'eta_aileron': float(1 - I['b_ail_b_wing']),
            'eta_slat': float(I['b_slat_b_wing']),
            'geometria': {k: float(v) for k, v in G.items()
                          if isinstance(v, (int, float))}}


# ------------------------------------------------------------------- saidas

def escreve_saidas(dados):
    req = dados['requisitos']
    casos = dados['casos']
    ws = dados['washouts']
    crit = dados['caso_critico']
    esc = dados['escolha_deg']
    esck = str(esc)
    eta_ail = dados['eta_aileron']
    CLp = dados['ponto_projeto']['CL']
    base, sel = casos[str(0.0)], casos[esck]
    cl_crit = [casos[str(w)][crit]['CLmax'] for w in ws]
    w_req = torcao_para(ws, cl_crit, req['clean_nec_TO'])
    w_alvo = torcao_para(ws, cl_crit, req['clean_alvo'])

    with open(os.path.join(OUT, 'q4_washout.json'), 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=2)

    # ---- tabela principal
    with open(os.path.join(OUT, 'tabela_washout.csv'), 'w', newline='',
              encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['washout_deg', 'it_fwd_deg', 'it_aft_deg',
                     'CLmax_fwd_livre', 'CLmax_fwd_trim', 'CLmax_aft_livre',
                     'CLmax_aft_trim', 'alpha_max_deg', 'eta_estol',
                     'eta_plat_ini', 'eta_plat_fim', 'CLmaxTO_disp',
                     'folga_TO', 'CD_cruzeiro_fwd', 'e_fwd',
                     'penal_CD_pct'])
        for w in ws:
            d = casos[str(w)]
            c = d[crit]
            disp = c['CLmax'] + req['dCL_TO']
            wr.writerow(['%+.1f' % w, '%.3f' % d['it_fwd'],
                         '%.3f' % d['it_aft'],
                         '%.4f' % d['fwd_livre']['CLmax'],
                         '%.4f' % d['fwd_trim']['CLmax'],
                         '%.4f' % d['aft_livre']['CLmax'],
                         '%.4f' % d['aft_trim']['CLmax'],
                         '%.2f' % c['alpha_max_deg'],
                         '%.4f' % c['eta_estol'],
                         '%.3f' % c['eta_plat_ini'],
                         '%.3f' % c['eta_plat_fim'],
                         '%.4f' % disp,
                         '%+.4f' % (disp - req['CLmaxTO_req']),
                         '%.6f' % d['cruz_fwd']['CD'],
                         '%.4f' % d['cruz_fwd']['e'],
                         '%+.2f' % (100*(d['cruz_fwd']['CD']
                                         / base['cruz_fwd']['CD'] - 1))])

    # ---- distribuicoes do caso escolhido e do caso base
    for rot, d in (('sem_washout', base), ('escolhido', sel)):
        dd = d[crit]['dist']
        with open(os.path.join(OUT, 'dist_%s.csv' % rot), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['eta', 'cl_norm', 'clmax_limite', 'razao'])
            for k in range(len(dd['eta'])):
                wr.writerow(['%.4f' % dd['eta'][k], '%.5f' % dd['cl_norm'][k],
                             '%.5f' % dd['clmax_lim'][k],
                             '%.5f' % dd['razao'][k]])

    # ---- tabelas LaTeX
    with open(os.path.join(TEX, 'tab_washout.tex'), 'w', encoding='utf-8') as f:
        f.write('\\begin{tabular}{rccccccc}\n\\toprule\n')
        f.write('$\\varepsilon_t$ [$^\\circ$] & $i_t$ (fwd) [$^\\circ$] & '
                '$\\alpha_{max}$ [$^\\circ$] & $C_{L\\,max}$ & '
                '$\\eta$ do estol & planalto $\\eta$ & '
                '$C_D$ (cruzeiro) & $e$ \\\\\n\\midrule\n')
        for w in ws:
            d = casos[str(w)]
            c = d[crit]
            f.write('$%+.0f$ & $%+.3f$ & %.2f & %.3f & %.3f & '
                    '%.2f--%.2f & %.5f & %.4f \\\\\n'
                    % (w, d['it_fwd'], c['alpha_max_deg'], c['CLmax'],
                       c['eta_estol'], c['eta_plat_ini'], c['eta_plat_fim'],
                       d['cruz_fwd']['CD'], d['cruz_fwd']['e']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(os.path.join(TEX, 'tab_washout_casos.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lcccccc}\n\\toprule\n')
        f.write('Caso & \\multicolumn{3}{c}{sem torcao} & '
                '\\multicolumn{3}{c}{$\\varepsilon_t = %+.0f^\\circ$} '
                '\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n' % esc)
        f.write('& $\\alpha_{max}$ & $C_{L\\,max}$ & $\\delta_e$ '
                '& $\\alpha_{max}$ & $C_{L\\,max}$ & $\\delta_e$ '
                '\\\\\n\\midrule\n')
        for nome, _, _, rotulo in CASOS:
            b, s = base[nome], sel[nome]
            f.write('%s & %.2f & %.3f & $%+.2f$ & %.2f & %.3f & $%+.2f$ '
                    '\\\\\n' % (rotulo, b['alpha_max_deg'], b['CLmax'],
                                b['delta_e_deg'], s['alpha_max_deg'],
                                s['CLmax'], s['delta_e_deg']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    with open(os.path.join(TEX, 'tab_washout_requisito.tex'), 'w',
              encoding='utf-8') as f:
        f.write('\\begin{tabular}{lccc}\n\\toprule\n')
        f.write('$\\varepsilon_t$ [$^\\circ$] & $C_{L\\,max}$ limpo (AVL) & '
                '$C_{L\\,max,TO}$ disponivel & folga sobre o requisito '
                '\\\\\n\\midrule\n')
        for w in ws:
            c = casos[str(w)][crit]
            disp = c['CLmax'] + req['dCL_TO']
            f.write('$%+.0f$ & %.3f & %.3f & $%+.3f$ \\\\\n'
                    % (w, c['CLmax'], disp, disp - req['CLmaxTO_req']))
        f.write('\\midrule\n')
        f.write('\\code{designTool} (Raymer) & %.3f & %.3f & $%+.3f$ '
                '\\\\\n' % (req['CLmax_clean_DT'], req['CLmaxTO_DT'],
                            req['CLmaxTO_DT'] - req['CLmaxTO_req']))
        f.write('\\bottomrule\n\\end{tabular}\n')

    # ---- figuras
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.2))
    a = axs[0]
    for nome, _, _, rot in CASOS:
        a.plot(ws, [casos[str(w)][nome]['CLmax'] for w in ws], 'o-', ms=4,
               lw=1.6, label=rot)
    a.axhline(req['clean_nec_TO'], color='k', ls='--', lw=1.3)
    a.annotate('necessario para decolar', (ws[-1], req['clean_nec_TO']),
               xytext=(4, 4), textcoords='offset points', fontsize=8)
    a.axhline(req['clean_alvo'], color='0.45', ls=':', lw=1.3)
    a.annotate('com %.0f%% de margem' % (100*req['margem']),
               (ws[-1], req['clean_alvo']), xytext=(4, -11),
               textcoords='offset points', fontsize=8, color='0.35')
    a.axvline(esc, color='tab:green', lw=1.2, alpha=0.6)
    a.set_xlabel(r'torção na ponta $\varepsilon_t$ [graus]')
    a.set_ylabel(r'$C_{L\,max}$ de asa limpa')
    a.set_title(r'(a) $C_{L\,max}$ vs. washout', fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, loc='lower left')

    a = axs[1]
    a.plot(ws, [casos[str(w)][crit]['eta_estol'] for w in ws], 'o-',
           color='tab:purple', ms=4, lw=1.8, label='estação crítica')
    a.fill_between(ws, [casos[str(w)][crit]['eta_plat_ini'] for w in ws],
                   [casos[str(w)][crit]['eta_plat_fim'] for w in ws],
                   color='tab:purple', alpha=0.18,
                   label=r'faixa com $c_\ell/c_{\ell\,max}>0,97$')
    a.axhline(eta_ail, color='tab:red', ls='--', lw=1.4)
    a.annotate('início do aileron', (ws[-1], eta_ail), xytext=(4, 4),
               textcoords='offset points', fontsize=8, color='tab:red')
    a.axvline(esc, color='tab:green', lw=1.2, alpha=0.6)
    a.set_xlabel(r'torção na ponta $\varepsilon_t$ [graus]')
    a.set_ylabel(r'$\eta = y/(b/2)$')
    a.set_title('(b) onde o estol começa', fontsize=10)
    a.set_ylim(0, 1)
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, loc='lower left')

    a = axs[2]
    cd = np.array([casos[str(w)]['cruz_fwd']['CD'] for w in ws])
    a.plot(ws, 1e4*cd, 'o-', color='tab:brown', ms=4, lw=1.8)
    a.set_xlabel(r'torção na ponta $\varepsilon_t$ [graus]')
    a.set_ylabel(r'$C_D$ de cruzeiro [contagens]')
    a2 = a.twinx()
    a2.plot(ws, [casos[str(w)]['cruz_fwd']['e'] for w in ws], 's--',
            color='tab:blue', ms=4, lw=1.6)
    a2.set_ylabel('fator de Oswald $e$', color='tab:blue')
    a2.tick_params(axis='y', colors='tab:blue')
    a.axvline(esc, color='tab:green', lw=1.2, alpha=0.6)
    a.set_title('(c) preço no cruzeiro', fontsize=10)
    a.grid(alpha=0.3)
    fig.suptitle('Efeito da torção geométrica na NJ-0502 (caso dimensionante: '
                 '%s)' % dict((c[0], c[3]) for c in CASOS)[crit], fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'washout_varredura.png'), dpi=200)
    plt.close(fig)

    # distribuicoes
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.7))
    mostrar = [w for w in ws if w in (0.0, -2.0, -4.0, esc, min(ws))]
    cores = plt.cm.viridis(np.linspace(0.05, 0.85, len(mostrar)))
    a = axs[0]
    d0 = casos[str(0.0)][crit]['dist']
    a.plot(d0['eta'], d0['clmax_lim'], 'k--', lw=1.8,
           label=r'$c_{\ell\,max}(Re_{local})$')
    for w, cor in zip(mostrar, cores):
        d = casos[str(w)][crit]
        a.plot(d['dist']['eta'], d['dist']['cl_norm'], color=cor, lw=2.2,
               label=r'$\varepsilon_t=%+.0f^\circ$: $C_{L\,max}=%.3f$'
                     % (w, d['CLmax']))
        a.plot(d['eta_estol'],
               np.interp(d['eta_estol'], d['dist']['eta'],
                         d['dist']['cl_norm']), 'o', color=cor, ms=8,
               mfc='white', mew=1.7)
    a.set_ylim(0, 1.95)
    a.set_ylabel(r'$c_\ell$ no plano normal')
    a.set_title('distribuição de sustentação no estol', fontsize=10)
    a = axs[1]
    for w, cor in zip(mostrar, cores):
        d = casos[str(w)][crit]
        a.plot(d['dist']['eta'], d['dist']['razao'], color=cor, lw=2.2,
               label=r'$\varepsilon_t=%+.0f^\circ$' % w)
    a.axhline(1.0, color='k', ls='--', lw=1.3)
    a.set_ylabel(r'$c_\ell/c_{\ell\,max}$')
    a.set_title('grau de carregamento', fontsize=10)
    for a in axs:
        a.axvspan(eta_ail, 1.0, color='0.9', zorder=0)
        a.annotate('faixa do aileron', (eta_ail + 0.01, 0.1), fontsize=8,
                   color='0.35')
        a.set_xlim(0, 1)
        a.set_xlabel(r'$\eta = y/(b/2)$')
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5, loc='lower left')
    fig.suptitle('Efeito do washout na distribuição de sustentação '
                 '(%s)' % dict((c[0], c[3]) for c in CASOS)[crit], fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'washout_distribuicoes.png'), dpi=200)
    plt.close(fig)

    # ---- item 5: polares com e sem washout
    pol = dados.get('polares')
    ajq5 = None
    if pol:
        ref = pol.get('referencia_q5')
        aj = pol['ajuste']
        ajq5 = ref['ajuste'] if ref else None

        with open(os.path.join(OUT, 'tabela_polares.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['caso', 'CD_sem_washout', 'CD_com_washout',
                         'dif_pct', 'LD_sem', 'LD_com', 'alpha_com',
                         'delta_e_com', 'CLmax_sem', 'CLmax_com'])
            for nome, _, _, _ in CASOS:
                d = pol['casos'][nome]
                r = ref['polares'][nome] if ref else None
                cd0 = r['CD_projeto'] if r else float('nan')
                wr.writerow([nome, '%.6f' % cd0, '%.6f' % d['CD_projeto'],
                             '%+.2f' % (100*(d['CD_projeto']/cd0 - 1)),
                             '%.2f' % (CLp/cd0),
                             '%.2f' % (CLp/d['CD_projeto']),
                             '%.3f' % d['alpha_projeto'],
                             '%+.3f' % d['delta_e_projeto'],
                             '%.4f' % (r['CLmax'] if r else float('nan')),
                             '%.4f' % d['CLmax']])

        with open(os.path.join(TEX, 'tab_washout_polar.tex'), 'w',
                  encoding='utf-8') as f:
            f.write('\\begin{tabular}{lcccccc}\n\\toprule\n')
            f.write('Configuracao & \\multicolumn{2}{c}{$C_D$ no ponto de '
                    'projeto} & & \\multicolumn{2}{c}{$C_L/C_D$} & '
                    '$\\alpha$ \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\n')
            f.write('& sem torcao & $\\varepsilon_t=%+.0f^\\circ$ & dif. & '
                    'sem torcao & com torcao & [$^\\circ$] '
                    '\\\\\n\\midrule\n' % esc)
            for nome, _, _, rotulo in CASOS:
                d = pol['casos'][nome]
                cd0 = ref['polares'][nome]['CD_projeto'] if ref else float('nan')
                f.write('%s & %.5f & %.5f & $%+.1f$\\%% & %.1f & %.1f & '
                        '%.2f \\\\\n'
                        % (rotulo, cd0, d['CD_projeto'],
                           100*(d['CD_projeto']/cd0 - 1), CLp/cd0,
                           CLp/d['CD_projeto'], d['alpha_projeto']))
            f.write('\\bottomrule\n\\end{tabular}\n')

        if ajq5:
            with open(os.path.join(TEX, 'tab_washout_ajuste.tex'), 'w',
                      encoding='utf-8') as f:
                f.write('\\begin{tabular}{lrr}\n\\toprule\n')
                f.write('Parametro & sem torcao & $\\varepsilon_t=%+.0f^\\circ$'
                        ' \\\\\n\\midrule\n' % esc)
                for lbl, k in (('$C_{D0}$', 'CD0'),
                               ('$C_{D\\alpha}$ [1/rad]', 'CD_alpha_por_rad'),
                               ('$C_{D\\alpha^2}$ [1/rad$^2$]',
                                'CD_alpha2_por_rad2'),
                               ('$R^2$', 'r2')):
                    f.write('%s & %.5f & %.5f \\\\\n'
                            % (lbl, ajq5[k], aj[k]))
                f.write('\\bottomrule\n\\end{tabular}\n')

        fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.8))
        a = axs[0]
        if ref:
            a.plot(ref['polar_designTool']['CD'], ref['polar_designTool']['CL'],
                   color='0.45', lw=1.3, ls=':', label='polar do designTool')
        for nome, _, _, rotulo in CASOS:
            d = pol['casos'][nome]
            if ref:
                r = ref['polares'][nome]
                a.plot(r['CD'], r['CL'], color='tab:red', lw=1.3, alpha=0.55)
            a.plot(d['CD'], d['CL'], color='tab:blue', lw=1.3, alpha=0.55)
        a.plot([], [], color='tab:red', lw=2.0, label='sem torção (item 5)')
        a.plot([], [], color='tab:blue', lw=2.0,
               label=r'$\varepsilon_t = %+.0f^\circ$' % esc)
        a.axhline(CLp, color='0.25', lw=0.9, ls='--')
        a.annotate('ponto de projeto', (0.055, CLp), xytext=(0, 4),
                   textcoords='offset points', fontsize=8, ha='right')
        a.set_xlabel(r'$C_D$')
        a.set_ylabel(r'$C_L$')
        a.set_title('polares completas (as quatro configurações)', fontsize=10)
        a = axs[1]
        for nome, _, _, rotulo in CASOS:
            d = pol['casos'][nome]
            if ref:
                r = ref['polares'][nome]
                a.plot(r['CD'], r['CL'], color='tab:red', lw=1.4, alpha=0.55)
                a.plot(r['CD_projeto'], CLp, 'o', color='tab:red', ms=7,
                       mfc='none', mew=1.6)
            a.plot(d['CD'], d['CL'], color='tab:blue', lw=1.4, alpha=0.55)
            a.plot(d['CD_projeto'], CLp, 'o', color='tab:blue', ms=7,
                   mfc='none', mew=1.6)
        a.set_xlim(0.014, 0.032)
        a.set_ylim(0.0, 0.9)
        a.set_xlabel(r'$C_D$')
        a.set_ylabel(r'$C_L$')
        a.set_title(r'detalhe no ponto de projeto ($C_L=%.3f$)' % CLp,
                    fontsize=10)
        for a in axs:
            a.grid(alpha=0.3)
        axs[0].legend(fontsize=8, loc='lower right')
        fig.suptitle('Item 5 revisitado: efeito do washout na polar de '
                     'arrasto ($M=%.2f$, $h=%.0f$ m)'
                     % (dados['ponto_projeto']['M'],
                        dados['ponto_projeto']['h_m']), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'washout_polares.png'), dpi=200)
        plt.close(fig)

    # ---- macros LaTeX
    c0, cs = base[crit], sel[crit]
    mac = {'woEsc': '%+.0f' % esc,
           'woEscAbs': '%.0f' % abs(esc),
           'woCont': '%.1f' % w_req,
           'woContAlvo': '%.1f' % w_alvo,
           'woRegra': dados['regra_escolha'],
           'woMargem': '%.0f' % (100*req['margem']),
           'woLista': ', '.join('$%+.0f^\\circ$' % w for w in ws),
           'woReqTO': '%.3f' % req['CLmaxTO_req'],
           'woReqLD': '%.3f' % req['CLmaxLD_req'],
           'woDeltaTO': '%.3f' % req['dCL_TO'],
           'woDeltaLD': '%.3f' % req['dCL_LD'],
           'woCleanNecTO': '%.3f' % req['clean_nec_TO'],
           'woCleanNecLD': '%.3f' % req['clean_nec_LD'],
           'woCleanAlvo': '%.3f' % req['clean_alvo'],
           'woCleanDT': '%.3f' % req['CLmax_clean_DT'],
           'woWS': '%.0f' % req['W0_S'],
           'woTW': '%.3f' % req['T0_W0'],
           'woPista': '%.0f' % req['s_TO'],
           'woErroLin': '%.0e' % dados['erro_linearidade'],
           'woEtaAil': '%.2f' % eta_ail,
           'woEtaSlat': '%.2f' % dados['eta_slat'],
           'woCLmaxZero': '%.3f' % c0['CLmax'],
           'woCLmaxEsc': '%.3f' % cs['CLmax'],
           'woGanhoPct': '%.0f' % (100*(cs['CLmax']/c0['CLmax'] - 1)),
           'woAlphaZero': '%.2f' % c0['alpha_max_deg'],
           'woAlphaEsc': '%.2f' % cs['alpha_max_deg'],
           'woEtaZero': '%.3f' % c0['eta_estol'],
           'woEtaEsc': '%.3f' % cs['eta_estol'],
           'woPlatZeroIni': '%.2f' % c0['eta_plat_ini'],
           'woPlatZeroFim': '%.2f' % c0['eta_plat_fim'],
           'woPlatEscIni': '%.2f' % cs['eta_plat_ini'],
           'woPlatEscFim': '%.2f' % cs['eta_plat_fim'],
           'woTOZero': '%.3f' % (c0['CLmax'] + req['dCL_TO']),
           'woTOEsc': '%.3f' % (cs['CLmax'] + req['dCL_TO']),
           'woFolgaZero': '%+.3f' % (c0['CLmax'] + req['dCL_TO']
                                     - req['CLmaxTO_req']),
           'woFolgaEsc': '%+.3f' % (cs['CLmax'] + req['dCL_TO']
                                    - req['CLmaxTO_req']),
           'woItFwdZero': '%+.3f' % base['it_fwd'],
           'woItAftZero': '%+.3f' % base['it_aft'],
           'woItFwdEsc': '%+.3f' % sel['it_fwd'],
           'woItAftEsc': '%+.3f' % sel['it_aft'],
           'woCDZero': '%.5f' % base['cruz_fwd']['CD'],
           'woCDEsc': '%.5f' % sel['cruz_fwd']['CD'],
           'woCDPct': '%+.1f' % (100*(sel['cruz_fwd']['CD']
                                      / base['cruz_fwd']['CD'] - 1)),
           'woCDCont': '%+.1f' % (1e4*(sel['cruz_fwd']['CD']
                                       - base['cruz_fwd']['CD'])),
           'woEZero': '%.4f' % base['cruz_fwd']['e'],
           'woEEsc': '%.4f' % sel['cruz_fwd']['e'],
           'woEfZero': '%.1f' % (dados['ponto_projeto']['CL']
                                 / base['cruz_fwd']['CD']),
           'woEfEsc': '%.1f' % (dados['ponto_projeto']['CL']
                                / sel['cruz_fwd']['CD'])}
    suf = {'fwd_livre': 'FwdLivre', 'fwd_trim': 'FwdTrim',
           'aft_livre': 'AftLivre', 'aft_trim': 'AftTrim'}
    for nome, s in suf.items():
        mac['woCLmaxEsc' + s] = '%.3f' % sel[nome]['CLmax']
        mac['woDeEsc' + s] = '%+.2f' % sel[nome]['delta_e_deg']
        mac['woAlphaEsc' + s] = '%.2f' % sel[nome]['alpha_max_deg']
    if pol:
        for nome, s in suf.items():
            d = pol['casos'][nome]
            mac['woCDPol' + s] = '%.5f' % d['CD_projeto']
            mac['woEfPol' + s] = '%.1f' % (CLp/d['CD_projeto'])
            mac['woAlphaPol' + s] = '%.2f' % d['alpha_projeto']
            if ref:
                cd0 = ref['polares'][nome]['CD_projeto']
                mac['woCDPolZero' + s] = '%.5f' % cd0
                mac['woCDPolDif' + s] = '%+.1f' % (100*(d['CD_projeto']/cd0 - 1))
        mac['woFitCDzero'] = '%.5f' % pol['ajuste']['CD0']
        mac['woFitCDa'] = '%.5f' % pol['ajuste']['CD_alpha_por_rad']
        mac['woFitCDaa'] = '%.5f' % pol['ajuste']['CD_alpha2_por_rad2']
        mac['woFitRdois'] = '%.5f' % pol['ajuste']['r2']
        if ajq5:
            mac['woFitCDzeroQ'] = '%.5f' % ajq5['CD0']
            mac['woFitCDaQ'] = '%.5f' % ajq5['CD_alpha_por_rad']
            mac['woFitCDaaQ'] = '%.5f' % ajq5['CD_alpha2_por_rad2']
    cds = np.array([casos[str(w)]['cruz_fwd']['CD'] for w in ws])
    ef = np.array([casos[str(w)]['cruz_fwd']['e'] for w in ws])
    imin, imax = int(np.argmin(cds)), int(np.argmax(ef))
    mac.update({'woWCDMin': '%+.0f' % ws[imin], 'woCDMin': '%.5f' % cds[imin],
                'woCDMinPct': '%+.1f' % (100*(cds[imin]/cds[0] - 1)),
                'woWEMax': '%+.0f' % ws[imax], 'woEMax': '%.4f' % ef[imax]})
    # providecommand para que as duas secoes (itens 4 e 5) possam dar \input
    with open(os.path.join(TEX, 'macros.tex'), 'w', encoding='utf-8') as f:
        f.write('% gerado por q4_washout.py\n')
        for k in sorted(mac):
            f.write('\\providecommand{\\%s}{%s}\n' % (k, mac[k]))

    print('\narquivos gravados em', OUT)
    print('tabelas LaTeX em', TEX)


def main(tex_only=False):
    for d in (OUT, TEX):
        os.makedirs(d, exist_ok=True)
    caminho = os.path.join(OUT, 'q4_washout.json')
    if tex_only:
        with open(caminho, encoding='utf-8') as f:
            dados = json.load(f)
        print('reaproveitando', caminho)
    else:
        dados = calcula()
    escreve_saidas(dados)


if __name__ == '__main__':
    main(tex_only=('--tex' in sys.argv))
