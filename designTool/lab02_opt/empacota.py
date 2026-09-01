'''
INSTITUTO TECNOLÓGICO DE AERONÁUTICA
PRJ-23 - Homework 02 - Grupo NJ-0502

Monta o .zip de entrega exigido pelo roteiro:
    NJ-0502_PRJ23_Lab02.zip

Inclui o relatório compilado, os fontes LaTeX, todos os códigos das três
seções e os resultados (CSV e figuras) que os scripts produziram.
Arquivos temporários (__pycache__, .log, .aux) ficam de fora.

Uso:  python empacota.py     (depois de compila.py)
'''

# IMPORTS
import os
import zipfile

# =========================================

_HERE = os.path.dirname(os.path.abspath(__file__))
NOME_ZIP = 'NJ-0502_PRJ23_Lab02.zip'

SECOES = [
    'otimizacao_aeronave_padrao',
    'otimizacao_NJ0502',
    'otimizacao_multiobjetivo',
]

# Extensões e nomes descartados.
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


# =========================================

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
            z.write(os.path.join(_HERE, nome), os.path.join('lab02_opt', nome))
            n += 1

        # Árvore Overleaf: tabelas e figuras no mesmo nível que main.tex.
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
                        z.write(completo, os.path.join('lab02_opt', rel))
                        n += 1

        # Códigos Python (permanecem nas pastas das seções).
        for secao in SECOES:
            base = os.path.join(_HERE, secao)
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in DIR_FORA]
                for fn in filenames:
                    completo = os.path.join(dirpath, fn)
                    rel = os.path.relpath(completo, _HERE)
                    if not inclui(rel):
                        continue
                    z.write(completo, os.path.join('lab02_opt', rel))
                    n += 1

    tam = os.path.getsize(destino)/1024.0/1024.0
    print('  gravado: %s' % destino)
    print('  %d arquivos, %.2f MB' % (n, tam))
