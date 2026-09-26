'''
Converte em PNG os graficos PostScript gerados pelo AVL (Xplot11, comando 'h'),
sem precisar de Ghostscript: interpreta os poucos operadores de desenho do
Xplot11 (M, L, CPSM, CO, SL, SG, setdash) e redesenha com matplotlib.

Uso: python ps2png.py arquivo.ps [arquivo.png] [--escuro]
  --escuro  fundo preto como na janela do AVL (linhas pretas viram brancas)
'''

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def le_ps(path):
    '''Devolve lista de (segmentos, cor, espessura, tracejado) na ordem do .ps.'''
    tokens = []
    with open(path) as f:
        for linha in f:
            linha = linha.split('%', 1)[0]
            if linha.lstrip().startswith('/'):
                continue
            tokens += linha.replace('[', ' [ ').replace(']', ' ] ').split()

    tracos = []
    pilha, caminho, polilinha = [], [], []
    cor, larg, dash = (0.0, 0.0, 0.0), 0.25, None
    em_array, array = False, []

    def fecha_polilinha():
        if len(polilinha) > 1:
            caminho.append(list(polilinha))
        polilinha.clear()

    def stroke():
        fecha_polilinha()
        if caminho:
            tracos.append((list(caminho), cor, larg, dash))
        caminho.clear()

    for t in tokens:
        if em_array:
            if t == ']':
                em_array = False
                pilha.append(array)
            else:
                array.append(float(t))
            continue
        if t == '[':
            em_array, array = True, []
            continue
        try:
            pilha.append(float(t))
            continue
        except ValueError:
            pass
        if t == 'M':
            y, x = pilha.pop()/10, pilha.pop()/10
            fecha_polilinha()
            polilinha.append((x, y))
        elif t == 'L':
            y, x = pilha.pop()/10, pilha.pop()/10
            polilinha.append((x, y))
        elif t in ('CPSM', 'stroke'):
            stroke()
        elif t == 'CO':
            stroke()
            b, g, r = pilha.pop(), pilha.pop(), pilha.pop()
            cor = (r/255, g/255, b/255)
        elif t == 'SL':
            stroke()
            larg = pilha.pop()
        elif t == 'SG':
            stroke()
            v = pilha.pop()
            cor = (v, v, v)
        elif t == 'setdash':
            stroke()
            pilha.pop()
            a = pilha.pop()
            dash = a if a else None
        else:
            pilha.clear()
    stroke()
    return tracos


def converte(ps, png, escuro=False, dpi=200):
    tracos = le_ps(ps)
    fundo = 'black' if escuro else 'white'
    fig = plt.figure(figsize=(11, 7.5), facecolor=fundo)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(fundo)
    xs, ys = [], []
    for segs, cor, larg, dash in tracos:
        if escuro and cor == (0.0, 0.0, 0.0):
            cor = (1.0, 1.0, 1.0)
        estilo = (0, tuple(dash)) if dash else 'solid'
        ax.add_collection(LineCollection(segs, colors=[cor],
                                         linewidths=max(larg, 0.5)*0.8,
                                         linestyles=[estilo]))
        for s in segs:
            xs += [p[0] for p in s]
            ys += [p[1] for p in s]
    m = 10
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(png, dpi=dpi, facecolor=fundo)
    plt.close(fig)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ps = args[0]
    png = args[1] if len(args) > 1 else ps.rsplit('.', 1)[0] + '.png'
    converte(ps, png, escuro='--escuro' in sys.argv)
    print('  gravado:', png)
