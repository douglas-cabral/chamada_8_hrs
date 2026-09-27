'''
PRJ-23 Lab 04 - Driver do AVL em modo batch e leitores das saidas ft/fs/st/sb.

Uma chamada de run_cases() abre uma unica sessao do AVL337 para um arquivo
.avl, fixa o Mach e a incidencia de empenagem (variavel de projeto "it" do
menu Design Changes) e executa varios casos, gravando as saidas em arquivos
texto (o AVL nao escreve nada em stdout que seja facil de casar com o caso).

Convencao dos casos:
    {'alpha': 5.0}                 -> a a 5       (profundor travado em zero)
    {'CL': 0.4710}                 -> a c 0.4710  (profundor travado em zero)
    {..., 'trim': True}            -> d4 pm 0.0   (arfagem compensada, CM = 0)
Sem 'trim' o profundor fica travado em zero (d4 d4 0), que e a condicao
"sem deflexoes de superficie de controle" pedida no enunciado.
'''

import os
import re
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AVLDIR = os.path.abspath(os.path.join(HERE, '..', 'AVL_package'))
AVL_EXE = os.path.join(AVLDIR, 'avl337.exe')

# Colunas da tabela "Strip Forces referred to Strip Area, Chord" do comando fs
FS_COLS = ['j', 'Yle', 'Chord', 'Area', 'c_cl', 'ai', 'cl_norm', 'cl',
           'cd', 'cdv', 'cm_c4', 'cm_LE', 'CP_x_c']


# ---------------------------------------------------------------- execucao

def _script(avl_file, mach, it, cases, tag):
    '''Monta o roteiro de teclas do AVL para uma sessao.'''
    c = ['load %s' % avl_file, 'oper', 'm', 'mn %.6f' % mach, '']
    if it is not None:
        c += ['de', '1 %.6f' % it, '']
    for k, case in enumerate(cases):
        if 'alpha' in case:
            c.append('a a %.6f' % case['alpha'])
        elif 'CL' in case:
            c.append('a c %.6f' % case['CL'])
        else:
            raise ValueError('caso sem alpha nem CL: %r' % case)
        c.append('d4 pm 0.0' if case.get('trim') else 'd4 d4 0.0')
        # 'ft'/'fs' seguidos de um nome de arquivo gravam em disco e voltam
        # sozinhos ao menu OPER (nao pode haver ENTER extra depois do nome)
        c += ['x',
              'ft', '%s_ft_%03d.txt' % (tag, k),
              'fs', '%s_fs_%03d.txt' % (tag, k)]
    c += ['', 'quit', '']
    return '\n'.join(c)


def run_cases(avl_file, mach, cases, it=None, tag='run', workdir=None,
              timeout=1200, keep=False):
    '''Executa os casos e devolve [{'ft':..., 'fs':...}, ...] na ordem dada.'''
    # o PID entra no caminho para que duas execucoes simultaneas nao apaguem
    # os arquivos uma da outra ao limpar a pasta no fim
    wd = workdir or os.path.join(HERE, '_avl_tmp', '%d_%s' % (os.getpid(), tag))
    os.makedirs(wd, exist_ok=True)
    # o AVL precisa achar o .avl, o .dat do perfil e o .dat da fuselagem
    for name in os.listdir(AVLDIR):
        if name.endswith(('.avl', '.dat', '.exe')):
            dst = os.path.join(wd, name)
            if not os.path.exists(dst) or os.path.getmtime(
                    os.path.join(AVLDIR, name)) > os.path.getmtime(dst):
                shutil.copy2(os.path.join(AVLDIR, name), dst)
    for k in range(len(cases)):
        for kind in ('ft', 'fs'):
            f = os.path.join(wd, '%s_%s_%03d.txt' % (tag, kind, k))
            if os.path.exists(f):
                os.remove(f)

    proc = subprocess.run([os.path.join(wd, 'avl337.exe')],
                          input=_script(avl_file, mach, it, cases, tag),
                          capture_output=True, text=True, cwd=wd,
                          timeout=timeout)

    out = []
    for k in range(len(cases)):
        ftf = os.path.join(wd, '%s_ft_%03d.txt' % (tag, k))
        fsf = os.path.join(wd, '%s_fs_%03d.txt' % (tag, k))
        if not (os.path.exists(ftf) and os.path.exists(fsf)):
            raise RuntimeError('AVL nao gerou as saidas do caso %d (%r)\n%s'
                               % (k, cases[k], proc.stdout[-2500:]))
        res = read_ft(ftf)
        res['fs'] = read_fs(fsf)
        res['caso'] = cases[k]
        out.append(res)
    if not keep:
        shutil.rmtree(wd, ignore_errors=True)
    return out


# ---------------------------------------------------------------- leitores

_NUM = r'([-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?)'


def _grab(text, key):
    m = re.search(re.escape(key) + r'\s*=\s*' + _NUM, text)
    return float(m.group(1).replace('D', 'E')) if m else float('nan')


def read_ft(path_or_text):
    '''Le a saida do comando ft (forcas totais).'''
    t = (open(path_or_text, encoding='utf-8', errors='replace').read()
         if os.path.exists(str(path_or_text)) else path_or_text)
    r = {k: _grab(t, k) for k in
         ('Alpha', 'Beta', 'Mach', 'CXtot', 'CYtot', 'CZtot', 'Cltot',
          'Cmtot', 'Cntot', 'CLtot', 'CDtot', 'CDvis', 'CDind',
          'CLff', 'CDff', 'e', 'Sref', 'Cref', 'Bref', 'Xref')}
    for name in ('flap', 'slat', 'aileron', 'elevator', 'rudder', 'it'):
        m = re.search(r'^\s*' + name + r'\s*=\s*' + _NUM, t, re.M)
        r[name] = float(m.group(1)) if m else float('nan')
    r['raw'] = t
    return r


def read_fs(path_or_text):
    '''Le a saida do comando fs. Devolve {nome_da_superficie: dict de arrays}.'''
    t = (open(path_or_text, encoding='utf-8', errors='replace').read()
         if os.path.exists(str(path_or_text)) else path_or_text)
    surfaces, cur, rows = {}, None, []

    def flush():
        if cur is not None:
            a = np.array(rows, dtype=float).reshape(-1, len(FS_COLS))
            d = {c: a[:, i] for i, c in enumerate(FS_COLS)}
            if cur in surfaces:   # metade espelhada pelo YDUPLICATE
                for c in FS_COLS:
                    surfaces[cur][c] = np.concatenate([surfaces[cur][c], d[c]])
            else:
                surfaces[cur] = d

    for line in t.splitlines():
        m = re.match(r'\s*Surface #\s*\d+\s+(\S.*?)\s*$', line)
        if m:
            flush()
            cur, rows = m.group(1).strip(), []
            continue
        f = line.split()
        # com cl de faixa proximo de zero o AVL deixa a ultima coluna
        # (C.P.x/c) em branco; a linha vem com uma coluna a menos
        if cur is not None and len(FS_COLS) - 1 <= len(f) <= len(FS_COLS):
            try:
                v = [float(x) for x in f]
            except ValueError:
                continue
            rows.append(v + [float('nan')]*(len(FS_COLS) - len(v)))
    flush()
    return surfaces


def wing_strips(fs, nome='Wing'):
    '''Semi-asa direita (Yle >= 0) ordenada por Yle.'''
    d = fs[nome]
    m = d['Yle'] >= -1e-9
    o = np.argsort(d['Yle'][m])
    return {c: d[c][m][o] for c in FS_COLS}


def read_derivs(path_or_text):
    '''Le pares "nome = valor" de uma saida st/sb (derivadas de estabilidade).'''
    t = (open(path_or_text, encoding='utf-8', errors='replace').read()
         if os.path.exists(str(path_or_text)) else path_or_text)
    d = {}
    for m in re.finditer(r'([A-Za-z][\w\'\.]*)\s*=\s*' + _NUM, t):
        d.setdefault(m.group(1), float(m.group(2)))
    return d
