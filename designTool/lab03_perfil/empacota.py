'''
PRJ-23 - Homework 03 - Grupo NJ-0502

Gera um .zip com o relatorio, os fontes e os resultados do laboratorio.

Uso: python empacota.py
'''

import os
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
NOME_ZIP = 'NJ-0502_PRJ23_Lab03.zip'
SECOES = ['airfoil_mod', 'analise_xfoil', 'eulerblock']

EXT_FORA = {'.pyc', '.log', '.aux', '.out', '.toc', '.fls', '.fdb_latexmk',
            '.synctex.gz'}
DIR_FORA = {'__pycache__', 'build', '.git'}


def inclui(caminho):
    partes = set(caminho.split(os.sep))
    if partes & DIR_FORA:
        return False
    ext = os.path.splitext(caminho)[1]
    if ext in EXT_FORA:
        return False
    return True


if __name__ == '__main__':
    destino = os.path.join(_HERE, NOME_ZIP)
    if os.path.exists(destino):
        os.remove(destino)

    itens = ['main.tex', 'compila.py', 'empacota.py', 'relatorio.pdf']
    itens += [secao + '.tex' for secao in SECOES]
    raiz = []
    for nome in itens:
        p = os.path.join(_HERE, nome)
        if os.path.isfile(p):
            raiz.append(nome)

    n = 0
    with zipfile.ZipFile(destino, 'w', zipfile.ZIP_DEFLATED) as z:
        for nome in raiz:
            z.write(os.path.join(_HERE, nome), os.path.join('lab03_perfil', nome))
            n += 1

        for secao in SECOES:
            for prefixo in ('tex_', 'resultados_'):
                pasta = os.path.join(_HERE, prefixo + secao)
                if not os.path.isdir(pasta):
                    continue
                for dirpath, dirnames, filenames in os.walk(pasta):
                    dirnames[:] = [d for d in dirnames if d not in DIR_FORA]
                    for fn in filenames:
                        completo = os.path.join(dirpath, fn)
                        rel = os.path.relpath(completo, _HERE)
                        if not inclui(rel):
                            continue
                        z.write(completo, os.path.join('lab03_perfil', rel))
                        n += 1

    tam = os.path.getsize(destino)/1024.0/1024.0
    print('  gravado: %s' % destino)
    print('  %d arquivos, %.2f MB' % (n, tam))
