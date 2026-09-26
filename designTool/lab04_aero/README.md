# Lab 04 — Análise aerodinâmica (AVL)

## Item 3: incidência da EH que zera o profundor no cruzeiro

Ponto de projeto (mesmo método do Lab 03, meio do cruzeiro, `my_airplane` atual):

| W0 [N] | W [N] | h [m] | ρ [kg/m³] | a [m/s] | M | V [m/s] | C_L | S_ref [m²] |
|---|---|---|---|---|---|---|---|---|
| 2 680 876 | 2 176 816 | 10 668 | 0,3805 | 296,59 | 0,85 | 252,10 | 0,4702 | 382,93 |

Resultado com a asa a `ANGLE 4.5` (`fwd.avl` / `aft.avl`):

| Caso | Xref [m] | **i_t** | α | δe | C_D | e |
|---|---|---|---|---|---|---|
| CG dianteiro (`fwd.avl`) | 28,751 | **−3,429°** | 0,175° | −0,003° | 0,02081 | 0,673 |
| CG traseiro (`aft.avl`) | 29,910 | **−1,959°** | 0,036° | 0,000° | 0,02023 | 0,756 |

A incidência de 4,5° da asa foi escolhida para a fuselagem voar nivelada (|α| < 0,2°) no ponto de projeto.

## Alterações em `AVL_package/fwd.avl` e `aft.avl`

- `ANGLE 4.5` na superfície `Wing` (incidência da asa em relação à fuselagem).
- `DESIGN it 1.0` nas duas seções da `Stab`. Como `it` é a única variável de projeto, no menu `de` ela é a **variável 1** (no `b737mod.avl` do professor é a 2).

## Rodando o AVL à mão

Abra o `avl337.exe` **de dentro de `AVL_package/`**; o `load` com caminho completo falha por causa dos espaços no caminho. **Feche e reabra o AVL antes de cada `load`**: recarregar na mesma sessão aplica o `it` duas vezes.

```
load fwd.avl
oper
m
mn 0.85
                (ENTER em branco: volta ao OPER)
a c 0.4702
d4 pm 0.0
de
1 -3.429        (aft.avl: 1 -1.959)
                (ENTER em branco)
x
t               (plano de Trefftz; h exporta plot.ps)
```

## Scripts

| Arquivo | Função |
|---|---|
| `q3_incidencia_eh.py` | Ajusta `it` pelo método da secante até \|δe\| < 0,01° e gera tudo em `resultados_q3/` |
| `ps2png.py` | Converte o `plot.ps` do AVL em PNG sem Ghostscript (`python ps2png.py arquivo.ps [--escuro]`) |

Uso: `python q3_incidencia_eh.py` (Windows, por causa do `avl337.exe`).

`resultados_q3/`: `trefftz_avl_<caso>.png` (figura do AVL, para o relatório), `trefftz_avl_<caso>_escuro.png`, `trefftz_<caso>.png` (redesenho a partir do `fs`), `q3_incidencia_eh.csv`, `ft_<caso>.txt`, `fs_<caso>.txt` e o `.ps` original.
