'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Grupo NJ-0502

Monta a árvore "achatada" que o main.tex espera e compila o PDF.

No repositório cada seção mora na sua própria pasta, com os códigos ao
lado das tabelas e figuras que produz. No Overleaf, porém, o main.tex faz
\input{<secao>} na raiz, e as tabelas/figuras são referenciadas como
tex_<secao>/... e resultados_<secao>/... a partir da raiz.

Este script copia tudo para build/ nesse formato e roda o pdflatex duas
vezes (para resolver as referências cruzadas).

Uso:  python compila.py
Saída: build/main.pdf  (e uma cópia em lab02_opt/relatorio.pdf)
'''

# IMPORTS
import os
import shutil
import subprocess
import sys

# =========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(_HERE, 'build')

SECOES = [
    'otimizacao_aeronave_padrao',
    'otimizacao_NJ0502',
    'otimizacao_multiobjetivo',
]

# =========================================


def monta_build():
    '''
    Recria build/ com o main.tex, os .tex das seções e as pastas
    tex_<secao>/ e resultados_<secao>/ todas no mesmo nível.
    '''
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    shutil.copy2(os.path.join(_HERE, 'main.tex'),
                 os.path.join(BUILD_DIR, 'main.tex'))

    for secao in SECOES:
        origem = os.path.join(_HERE, secao)

        tex = os.path.join(origem, secao + '.tex')
        if not os.path.isfile(tex):
            raise RuntimeError('não encontrei %s' % tex)
        shutil.copy2(tex, os.path.join(BUILD_DIR, secao + '.tex'))

        for prefixo in ('tex_', 'resultados_'):
            pasta = os.path.join(origem, prefixo + secao)
            if os.path.isdir(pasta):
                shutil.copytree(pasta,
                                os.path.join(BUILD_DIR, prefixo + secao))
        print('  seção copiada: %s' % secao)


def compila():
    '''
    Roda o pdflatex duas vezes dentro de build/.
    '''
    for passada in (1, 2):
        proc = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             'main.tex'],
            cwd=BUILD_DIR, capture_output=True, text=True,
            encoding='utf-8', errors='replace')
        print('  pdflatex passada %d: código %d' % (passada, proc.returncode))
        if proc.returncode != 0:
            saida = proc.stdout.splitlines()
            erros = [ln for ln in saida
                     if ln.startswith('!') or 'Error' in ln]
            print('\n'.join(erros[-40:]) or '\n'.join(saida[-40:]))
            return False
    return True


# =========================================

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
        print('  cópia em  : %s' % destino)
    else:
        print('\n  *** falha na compilação ***')
        sys.exit(1)
