'''
PRJ-23 Lab 04 - cl_max do perfil do Lab 03 em funcao do Reynolds local.

O metodo da secao critica compara o cl da secao (coluna cl_norm do comando fs
do AVL, que ja e o cl no plano normal ao bordo de ataque) com o cl_max do
perfil. Como a corda varia de 10,02 m na raiz a 2,00 m na ponta, o Reynolds
local varia por um fator de 5 e o cl_max nao e constante ao longo da
envergadura. Este modulo levanta cl_max(Re) com o XFOIL para o mesmo perfil
gravado em AVL_package/airfoil_lab03.dat (o otimo do Lab 03) e devolve um
interpolador.

O driver e o mesmo usado na campanha do Lab 03: PPAR com T=0,45 e N=200,
varredura em alpha com retomada de passo menor quando o XFOIL perde a
convergencia antes de caracterizar o estol.

Uso: python xfoil_perfil.py  (grava resultados_comum/clmax_perfil_Re.json)
'''

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AVLDIR = os.path.abspath(os.path.join(HERE, '..', 'AVL_package'))
XFOIL_EXE = os.path.abspath(os.path.join(
    HERE, '..', 'lab03_perfil', 'analise_xfoil', 'xfoil.exe'))
PERFIL = os.path.join(AVLDIR, 'airfoil_lab03.dat')
RESDIR = os.path.join(HERE, 'resultados_comum')

# 'corrente': o perfil como esta no .avl (secao paralela ao escoamento);
# 'normal'  : o mesmo perfil visto no plano normal ao bordo de ataque da asa
#             enflechada - espessura relativa esticada por 1/cos(Lambda), que e
#             o perfil coerente com a teoria de enflechamento simples embutida
#             na coluna cl_norm do AVL.
CACHES = {'corrente': os.path.join(RESDIR, 'clmax_perfil_Re.json'),
          'normal': os.path.join(RESDIR, 'clmax_perfil_Re_normal.json')}


def _dat_com_titulo(destino, titulo='AIRFOIL_LAB03', esticar_y=1.0):
    '''airfoil_lab03.dat nao tem linha de titulo; o XFOIL prefere que tenha.

    esticar_y > 1 gera o perfil do plano normal (t/c dividido por cos(Lambda)).
    '''
    xy = np.loadtxt(PERFIL)
    with open(destino, 'w', encoding='utf-8', newline='\n') as f:
        f.write(titulo + '\n')
        for x, y in xy:
            f.write('%13.9f %13.9f\n' % (x, esticar_y*y))
    return destino


def read_polar(path):
    rows, started = [], False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip().startswith('------'):
                started = True
                continue
            if started and line.strip():
                try:
                    rows.append([float(v) for v in line.split()[:7]])
                except ValueError:
                    pass
    return np.array(rows).reshape(-1, 7)


def run_polar(Re, mach, a0=0.0, a1=24.0, da=0.25, iters=200, npanel=200,
              te_bunch=0.45, workdir=None, timeout=240, esticar_y=1.0):
    own = workdir is None
    wd = workdir or tempfile.mkdtemp(prefix='xf_')
    os.makedirs(wd, exist_ok=True)
    _dat_com_titulo(os.path.join(wd, 'af.dat'), esticar_y=esticar_y)
    polar = os.path.join(wd, 'polar.txt')
    if os.path.exists(polar):
        os.remove(polar)
    cmds = ['PLOP', 'G', '',
            'LOAD af.dat',
            'PPAR', 'T', '%g' % te_bunch, 'N', '%d' % npanel, '', '',
            'OPER', 'ITER', '%d' % iters,
            'VISC', '%g' % Re, 'MACH', '%g' % mach,
            'PACC', 'polar.txt', '',
            'ASEQ', '%g' % a0, '%g' % a1, '%g' % da,
            'PACC', '', 'QUIT', '']
    try:
        proc = subprocess.run([XFOIL_EXE], input='\n'.join(cmds), text=True,
                              cwd=wd, capture_output=True, timeout=timeout)
        out = proc.stdout
    except subprocess.TimeoutExpired as e:
        out = str(e.stdout) + '\n*** TIMEOUT ***'
    data = read_polar(polar) if os.path.exists(polar) else np.zeros((0, 7))
    if own:
        shutil.rmtree(wd, ignore_errors=True)
    return data, out


def merge_polars(a, b):
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    d = {round(r[0], 4): r for r in b}
    d.update({round(r[0], 4): r for r in a})
    return np.array([d[k] for k in sorted(d)])


def moving_median(y, k=5):
    n, h = len(y), k//2
    return np.array([np.median(y[max(0, i-h):min(n, i+h+1)]) for i in range(n)])


def summarize(data, fit_range=(0.0, 5.0)):
    if len(data) == 0:
        return {'clmax': float('nan'), 'clmax_mm5': float('nan'),
                'alpha_clmax': float('nan'), 'cla_per_rad': float('nan'),
                'alpha_zl': float('nan'), 'stall_captured': False, 'n': 0}
    alpha, cl = data[:, 0], data[:, 1]
    i = int(np.argmax(cl))
    after = cl[i+1:]
    stall = bool(len(after) >= 2 and np.min(after) < cl[i] - 0.02)
    m = (alpha >= fit_range[0]) & (alpha <= fit_range[1])
    if m.sum() >= 3:
        p = np.polyfit(alpha[m], cl[m], 1)
        cla, a0l = p[0], -p[1]/p[0]
    else:
        cla, a0l = float('nan'), float('nan')
    return {'clmax': float(cl[i]),
            'clmax_mm5': float(np.max(moving_median(cl, 5))) if len(cl) >= 5
            else float(cl[i]),
            'alpha_clmax': float(alpha[i]), 'cla_per_deg': float(cla),
            'cla_per_rad': float(cla*180/np.pi), 'alpha_zl': float(a0l),
            'stall_captured': stall, 'n': int(len(alpha))}


def run_clmax(Re, mach, a0=0.0, a1=24.0, da=0.25, retries=3, npanel=200,
              workdir=None, esticar_y=1.0):
    own = workdir is None
    base = workdir or tempfile.mkdtemp(prefix='xfc_')
    os.makedirs(base, exist_ok=True)
    data, out = run_polar(Re, mach, a0, a1, da, npanel=npanel,
                          workdir=os.path.join(base, 'sweep0'),
                          esticar_y=esticar_y)
    for k in range(retries):
        s = summarize(data)
        if s['n'] == 0 or s['stall_captured'] or '*** TIMEOUT ***' in out:
            break
        a_last = float(data[:, 0].max())
        if a_last >= a1 - 1e-9:
            break
        start = max(a0, a_last - 1.0 - k*0.5)
        new, out = run_polar(Re, mach, start, min(a1, a_last + 3.0), da/2,
                             npanel=npanel, esticar_y=esticar_y,
                             workdir=os.path.join(base, 'retry%d' % (k+1)))
        if len(new) == 0 or new[:, 0].max() <= a_last + 1e-9:
            break
        data = merge_polars(data, new)
    if own:
        shutil.rmtree(base, ignore_errors=True)
    return data


def levanta_clmax_Re(Res, mach, workdir=None, verbose=True, esticar_y=1.0):
    '''cl_max do perfil para uma lista de Reynolds, no mesmo Mach.'''
    out = []
    for Re in Res:
        wd = (os.path.join(workdir, 'Re%.3e' % Re) if workdir else None)
        data = run_clmax(Re, mach, workdir=wd, esticar_y=esticar_y)
        s = summarize(data)
        s['Re'] = float(Re)
        s['mach'] = float(mach)
        out.append(s)
        if verbose:
            print('  Re=%9.3e  clmax=%6.4f (mm5 %6.4f)  alpha=%5.2f  '
                  'cla=%5.3f/rad  estol=%s'
                  % (Re, s['clmax'], s['clmax_mm5'], s['alpha_clmax'],
                     s['cla_per_rad'], s['stall_captured']))
    return out


def carrega(mach=None, Res=None, refaz=False, workdir=None, plano='corrente',
            esticar_y=1.0):
    '''Le o cache; levanta a curva se necessario. Devolve (dados, interpolador).'''
    os.makedirs(RESDIR, exist_ok=True)
    cache = CACHES[plano]
    dados = None
    if os.path.exists(cache) and not refaz:
        with open(cache, encoding='utf-8') as f:
            d = json.load(f)
        if mach is None or abs(d['mach'] - mach) < 1e-6:
            dados = d
    if dados is None:
        if mach is None or Res is None:
            raise RuntimeError('cache ausente (%s): rode xfoil_perfil.py'
                               % plano)
        casos = levanta_clmax_Re(Res, mach, workdir=workdir,
                                 esticar_y=esticar_y)
        dados = {'mach': float(mach), 'perfil': os.path.basename(PERFIL),
                 'plano': plano, 'esticar_y': float(esticar_y),
                 'casos': casos}
        with open(cache, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2)

    ok = [c for c in dados['casos'] if np.isfinite(c['clmax'])]
    ok.sort(key=lambda c: c['Re'])
    lR = np.log10([c['Re'] for c in ok])
    cm = np.array([c['clmax'] for c in ok])

    def clmax_de_Re(Re):
        '''Interpolacao linear em log(Re), extrapolacao por patamar.'''
        return np.interp(np.log10(np.atleast_1d(Re)), lR, cm)

    return dados, clmax_de_Re


def main():
    import ponto_projeto as pj
    import opt_common as oc
    ap = oc.run_designTool(oc.get_baseline())
    bv = pj.condicao_baixa_velocidade(ap)
    G, I = ap['geometry'], ap['inputs']
    cosL = float(np.cos(I['sweep_w']))
    # cordas que realmente ocorrem na asa (raiz -> ponta) + a MAC
    cordas = np.array([G['cr_w'], 8.0, G['cm_w'], 5.0, 4.0, 3.0, G['ct_w']])
    Res = np.sort(bv['Re_por_metro']*cordas)
    print('perfil %s   cordas [m]: %s'
          % (os.path.basename(PERFIL), np.round(np.sort(cordas), 3)))

    print('\n[1] plano da corrente livre: M=%.4f' % bv['M'])
    carrega(mach=bv['M'], Res=Res, refaz=True, plano='corrente',
            workdir=os.path.join(HERE, '_xfoil_tmp', 'corrente'))

    print('\n[2] plano normal ao bordo de ataque (Lambda=%.2f graus): '
          'M_n=%.4f, Re_n=Re*cos(Lambda), t/c esticado por %.4f'
          % (np.degrees(I['sweep_w']), bv['M']*cosL, 1/cosL))
    carrega(mach=bv['M']*cosL, Res=Res*cosL, refaz=True, plano='normal',
            esticar_y=1/cosL,
            workdir=os.path.join(HERE, '_xfoil_tmp', 'normal'))

    print('\ngravado em', ' e '.join(CACHES.values()))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.abspath(os.path.join(
        HERE, '..', 'lab02_opt', 'otimizacao_NJ0502')))
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
    main()
