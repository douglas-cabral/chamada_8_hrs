'''
PRJ-23 - Homework 03 - Grupo NJ-0502

Monta a arvore "achatada" que o main.tex espera e compila o PDF.

Uso: python compila.py
Saida: build/main.pdf e uma copia em lab03_perfil/relatorio.pdf
'''

import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')
SECOES = ['airfoil_mod', 'analise_xfoil', 'eulerblock']


def monta_build():
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    shutil.copy2(os.path.join(_HERE, 'main.tex'),
                 os.path.join(BUILD_DIR, 'main.tex'))

    for secao in SECOES:
        tex = os.path.join(_HERE, secao + '.tex')
        if not os.path.isfile(tex):
            raise RuntimeError('nao encontrei %s' % tex)
        shutil.copy2(tex, os.path.join(BUILD_DIR, secao + '.tex'))

        for prefixo in ('tex_', 'resultados_'):
            pasta = os.path.join(_HERE, prefixo + secao)
            if os.path.isdir(pasta):
                shutil.copytree(pasta, os.path.join(BUILD_DIR, prefixo + secao))
        print('  secao copiada: %s' % secao)


def compila():
    for passada in (1, 2):
        try:
            proc = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                cwd=BUILD_DIR, capture_output=True, text=True,
                encoding='utf-8', errors='replace')
        except FileNotFoundError:
            print('  pdflatex nao foi encontrado no PATH.')
            return False
        print('  pdflatex passada %d: codigo %d' % (passada, proc.returncode))
        if proc.returncode != 0:
            saida = proc.stdout.splitlines()
            erros = [ln for ln in saida if ln.startswith('!') or 'Error' in ln]
            print('\n'.join(erros[-40:]) or '\n'.join(saida[-40:]))
            return False
    return True


if __name__ == '__main__':
    print('=' * 70)
    print('  MONTANDO build/')
    print('=' * 70)
    monta_build()

    print('\n' + '=' * 70)
    print('  COMPILANDO')
    print('=' * 70)
    ok = compila()

    pdf = os.path.join(BUILD_DIR, 'main.pdf')
    if ok and os.path.isfile(pdf):
        destino = os.path.join(_HERE, 'relatorio.pdf')
        shutil.copy2(pdf, destino)
        print('\n  PDF gerado: %s' % pdf)
        print('  copia em  : %s' % destino)
    else:
        print('\n  *** falha na compilacao ***')
        sys.exit(1)
