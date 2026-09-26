'''
PRJ-23 Lab 04 - Item 3: incidencia da EH (variavel it) que anula a deflexao
de profundor no cruzeiro, para CG dianteiro (fwd.avl) e traseiro (aft.avl).

Roteiro da Tab. 2 do enunciado, automatizado: para cada arquivo, o AVL e
executado com CL = CL de projeto e Cm = 0 via profundor (d4 pm 0); o it e
atualizado pelo metodo da secante ate |delta_e| < TOL_DE.

Uso: python q3_incidencia_eh.py
Saidas em resultados_q3/:
  - q3_incidencia_eh.csv         tabela com it, alpha, delta_e, CL, CD, e
  - ft_<caso>.txt / fs_<caso>.txt  saida do AVL (forcas totais / por faixa)
  - trefftz_<caso>.ps            plano de Trefftz exportado pelo proprio AVL
  - trefftz_avl_<caso>.png       esse .ps convertido (fundo branco e _escuro)
  - trefftz_<caso>.png           plano de Trefftz redesenhado a partir do fs
'''

import csv
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ps2png

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

AVL_DIR = os.path.join(_ROOT, 'AVL_package')
AVL_EXE = os.path.join(AVL_DIR, 'avl337.exe')
RESULTS_DIR = os.path.join(_HERE, 'resultados_q3')

CASOS = [('fwd', 'fwd.avl', 'CG dianteiro'),
         ('aft', 'aft.avl', 'CG traseiro')]
IT_INDEX = 1        # 'it' e a unica variavel DESIGN dos arquivos .avl
TOL_DE = 0.01       # graus (o roteiro pede 0,1)
MAX_ITER = 20


def ponto_projeto():
    '''Mesmo ponto de projeto do Lab 03 (meio do cruzeiro), com o my_airplane atual.'''
    sys.path.insert(0, os.path.join(_ROOT, 'lab03_perfil', 'airfoil_mod'))
    import run_1_ponto_projeto
    return run_1_ponto_projeto.run()


def roda_avl(arquivo, mach, CL, it, extra=''):
    cmds = ['load %s' % arquivo, 'oper',
            'm', 'mn %.4f' % mach, '',
            'a c %.6f' % CL,
            'd4 pm 0.0',
            'de', '%d %.6f' % (IT_INDEX, it), '',
            'x', extra, '', 'quit']
    p = subprocess.run([AVL_EXE], input='\n'.join(cmds) + '\n', cwd=AVL_DIR,
                       capture_output=True, text=True, timeout=120)
    return p.stdout


def le_ft(saida):
    '''Le o ultimo bloco de forcas totais impresso pelo AVL.'''
    bloco = saida[saida.rfind('Vortex Lattice Output -- Total Forces'):]
    def val(nome):
        m = re.search(r'\b%s\s*=\s*(-?[\d.Ee+-]+)' % re.escape(nome), bloco)
        return float(m.group(1))
    return {k: val(k) for k in ['Alpha', 'CLtot', 'CDtot', 'CDvis', 'CDind',
                                'Cmtot', 'CLff', 'CDff', 'e', 'elevator', 'it']}


def ajusta_it(arquivo, mach, CL):
    it0, it1 = 0.0, -2.0
    de0 = le_ft(roda_avl(arquivo, mach, CL, it0))['elevator']
    hist = [(it0, de0)]
    for _ in range(MAX_ITER):
        r = le_ft(roda_avl(arquivo, mach, CL, it1))
        de1 = r['elevator']
        hist.append((it1, de1))
        if abs(de1) < TOL_DE:
            return it1, r, hist
        it0, it1, de0 = it1, it1 - de1*(it1 - it0)/(de1 - de0), de1
    raise RuntimeError('secante nao convergiu para %s' % arquivo)


def le_fs(saida):
    '''Faixas das superficies Wing e Stab (lado +y e YDUP) a partir do fs.'''
    faixas = {}
    atual = None
    for linha in saida.splitlines():
        m = re.match(r'\s*Surface #\s*\d+\s+(\w+)', linha)
        if m:
            atual = m.group(1) if m.group(1) in ('Wing', 'Stab') else None
            continue
        if atual is None:
            continue
        cols = linha.split()
        if len(cols) == 13 and cols[0].isdigit():
            y, c, ccl, cl = float(cols[1]), float(cols[2]), float(cols[4]), float(cols[7])
            faixas.setdefault(atual, []).append((y, c, ccl, cl))
    for k in faixas:
        faixas[k].sort()
    return faixas


def plota_trefftz(faixas, cref, titulo, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cores = {'Wing': '#1f77b4', 'Stab': '#d62728'}
    for sup, pts in faixas.items():
        y = [p[0] for p in pts]
        ax.plot(y, [p[2]/cref for p in pts], '-', color=cores[sup],
                label=r'$c\,c_\ell/c_{ref}$ ' + sup)
        ax.plot(y, [p[3] for p in pts], '--', color=cores[sup],
                label=r'$c_\ell$ ' + sup)
    ax.axhline(0.0, color='k', lw=0.5)
    ax.set_xlabel('y [m]')
    ax.set_ylabel(r'$c_\ell$, $c\,c_\ell/c_{ref}$')
    ax.set_title(titulo, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pp = ponto_projeto()
    mach, CL = pp['M'], pp['clref']
    print('Ponto de projeto: M = %.3f  CL = %.4f' % (mach, CL))

    linhas = []
    for tag, arquivo, nome in CASOS:
        it, r, hist = ajusta_it(arquivo, mach, CL)
        print('\n%s (%s): it = %.3f deg' % (nome, arquivo, it))
        for k, (i, d) in enumerate(hist):
            print('   iter %d: it = %8.4f  delta_e = %8.4f' % (k, i, d))

        ft = roda_avl(arquivo, mach, CL, it)
        with open(os.path.join(RESULTS_DIR, 'ft_%s.txt' % tag), 'w') as f:
            f.write(ft[ft.rfind('Vortex Lattice Output -- Total Forces'):])
        fs = roda_avl(arquivo, mach, CL, it, extra='fs')
        with open(os.path.join(RESULTS_DIR, 'fs_%s.txt' % tag), 'w') as f:
            f.write(fs[fs.find('Surface and Strip Forces'):])

        ps = os.path.join(AVL_DIR, 'plot.ps')
        if os.path.exists(ps):
            os.remove(ps)
        roda_avl(arquivo, mach, CL, it, extra='t\nh\n')
        if os.path.exists(ps):
            destino = os.path.join(RESULTS_DIR, 'trefftz_%s.ps' % tag)
            os.replace(ps, destino)
            base = os.path.join(RESULTS_DIR, 'trefftz_avl_%s' % tag)
            ps2png.converte(destino, base + '.png')
            ps2png.converte(destino, base + '_escuro.png', escuro=True)

        cref = float(re.search(r'Cref =\s*([\d.]+)', ft).group(1))
        titulo = (r'%s: $i_t$ = %.3f°, $\alpha$ = %.3f°, $\delta_e$ = %.4f°, '
                  r'$C_L$ = %.4f, $C_D$ = %.5f, e = %.4f'
                  % (nome, it, r['Alpha'], r['elevator'], r['CLtot'],
                     r['CDtot'], r['e']))
        plota_trefftz(le_fs(fs), cref, titulo,
                      os.path.join(RESULTS_DIR, 'trefftz_%s.png' % tag))

        linhas.append({'caso': tag, 'arquivo': arquivo, 'Mach': mach,
                       'CL_alvo': CL, 'it_deg': it, 'alpha_deg': r['Alpha'],
                       'delta_e_deg': r['elevator'], 'CLtot': r['CLtot'],
                       'CDtot': r['CDtot'], 'CDind': r['CDind'],
                       'Cmtot': r['Cmtot'], 'e': r['e'],
                       'n_iter': len(hist)})

    path = os.path.join(RESULTS_DIR, 'q3_incidencia_eh.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print('\n  gravado:', path)


if __name__ == '__main__':
    main()
