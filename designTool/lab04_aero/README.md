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

## Item 4: método da seção crítica (CLmax de asa limpa)

Condição: V2 = 1,2 Vs de decolagem ao nível do mar (M = 0,253, Re_MAC = 4,0e7),
que é o mesmo par (Re, M) em que o cl_max do perfil foi levantado no Lab 03.
Asa limpa (flap, slat e aileron neutros), `it` do item 3.

O limite de cl_max varia com a envergadura, porque a corda cai de 10,02 m na
raiz a 2,00 m na ponta: o XFOIL dá **1,783 na raiz e 1,526 na ponta**.
O critério é o do roteiro (passo 14 da Tab. 3): comparar a coluna `cl_norm` do
comando `fs` — o cl da seção no plano normal, igual a `cl`/cos²Λ — com esse
limite.

| Caso | i_t | α_max (fuselagem) | **C_Lmax** | δe | η do estol |
|---|---|---|---|---|---|
| CG dianteiro, sem trimagem | −3,429° | 5,98° | 0,864 | +0,00° | 0,907 |
| CG dianteiro, com trimagem | −3,429° | 5,99° | **0,854** | −1,95° | 0,907 |
| CG traseiro, sem trimagem | −1,959° | 5,98° | 0,877 | +0,00° | 0,907 |
| CG traseiro, com trimagem | −1,959° | 5,98° | 0,882 | +0,97° | 0,907 |

**Atenção:** com o `ANGLE 4.5` do item 3, o α do AVL é o ângulo da *fuselagem*;
a asa vê α + 4,5°, ou seja, 10,48° no estol.

**O estol começa em η = 0,91, dentro da faixa do aileron** (η 0,68 à ponta), e a
razão cl/cl_max fica acima de 0,97 em toda a faixa 0,79 ≤ η ≤ 0,93 — separação
abrupta e sem aviso. Causa: afilamento 0,2 com torção nula.

O C_Lmax obtido é 68% do `CLmax_clean = 0,9·cl_max·cosΛ = 1,304` do designTool.
A tabela de sensibilidade (`resultados_q4/tabela_criterios.csv`) mostra que o
valor do designTool fica no extremo otimista da faixa do método.

## Estudo de torção geométrica (washout) — entra nos itens 4 e 5

`q4_washout.py` varre torção linear de 0° na raiz a −1°…−8° na ponta. Para cada
valor **refaz a cadeia inteira**: recalcula o `it` do item 3 (a torção muda o
ângulo de sustentação nula e o momento da asa), relê o ponto de projeto de
cruzeiro e refaz o método da seção crítica nos quatro casos. O α de estol vem de
um modelo linear (α = 0° e 12°) corrigido por Newton sobre o AVL real, até
|max(cl/cl_max) − 1| < 2e−4 — o AVL não é exatamente linear em α.

| ε_t | i_t (fwd) | α_max | **C_Lmax** | η do estol | planalto η | C_D cruzeiro | e |
|---|---|---|---|---|---|---|---|
| 0° | −3,430° | 5,99° | 0,854 | 0,907 | 0,79–0,93 | 0,02081 | 0,673 |
| −2° | −2,771° | 7,42° | 0,926 | 0,846 | 0,75–0,93 | 0,02020 | 0,757 |
| −3° | −2,444° | 8,08° | 0,958 | 0,846 | 0,73–0,92 | **0,02009** | 0,788 |
| −5° | −1,792° | 9,41° | 1,021 | 0,773 | 0,65–0,91 | 0,02022 | **0,818** |
| **−7°** | **−1,143°** | **10,61°** | **1,074** | **0,773** | **0,50–0,88** | **0,02084** | **0,798** |
| −8° | −0,819° | 11,20° | 1,100 | 0,733 | 0,45–0,86 | 0,02133 | 0,771 |

**Critério de escolha:** o requisito de decolagem do próprio designTool,
`CLmaxTO ≥ 0,2387·(W0/S)/(σ·(T0/W0)·s_TO) = 1,988`. Somando o ΔCLmax = 0,921 de
flap+slat do Raymer, a asa limpa precisa de **1,066** — a asa reta entrega 0,854.
O pouso não é crítico (Torenbeek exige só 0,728 de asa limpa).

**Escolhido: ε_t = −7°** (menor da lista que fecha o requisito; contínuo: −6,7°).
Uma margem de 5% exigiria −9,9°, que não é realista — a leitura mistura o C_Lmax
conservador do AVL com um Δ empírico, e o recado é que **a margem de decolagem do
projeto é apertada**, não que a asa precisa de 10° de torção.

Três achados que valem para o relatório:

1. **A torção paga por si até ~−3°.** O C_D de cruzeiro *cai* 3,5% (mínimo
   0,02009 em −3°) e o e sobe de 0,673 para 0,818 (máximo em −5°): com λ = 0,2 e
   torção nula o carregamento está longe do elíptico. Em −7° o C_D volta ao valor
   da asa reta (+0,1% no CG dianteiro) — a torção necessária é **grátis em
   cruzeiro**.
2. **O CG traseiro perde a vantagem** (+3,9% de C_D). Ela vinha só da menor carga
   descendente na EH; com a torção o `it` do CG traseiro cruza zero entre −5° e
   −6° e a empenagem passa a sustentar, e aí o arrasto induzido dela mesma anula
   o alívio na asa.
3. **O planalto de carga alarga** (0,79–0,93 → 0,50–0,88). Maximizar C_Lmax e
   suavizar o estol são objetivos conflitantes: o ótimo de sustentação é a margem
   uniforme, que é justamente o estol mais abrupto. E mesmo em −8° a seção crítica
   não sai da faixa do aileron — isso exige mexer no afilamento, não na torção.

**Pendência:** adotar a torção obriga a rever o `i_w` do item 3. Com −7° a
fuselagem deixa de voar nivelada no ponto de projeto (α = 2,04° em vez de 0,18°),
o que pediria `ANGLE` de cerca de 6,5°.

Saídas em `resultados_washout/` e `tex_washout/`. Os `.avl` torcidos são gerados
e apagados dentro da execução; `fwd.avl` e `aft.avl` não são tocados.

## Item 5: polares de arrasto

Quatro polares em M = 0,85 e h = 10 668 m, de C_L = −0,5 até o C_Lmax do item 4.

| Configuração | α | δe | C_D (AVL) | C_D (designTool) | dif. |
|---|---|---|---|---|---|
| CG dianteiro, sem deflexões | 0,18° | +0,00° | 0,02081 | 0,02183 | −4,7% |
| CG traseiro, sem deflexões | 0,04° | +0,00° | 0,02023 | 0,02183 | −7,3% |
| CG dianteiro, compensada | 0,18° | +0,00° | 0,02081 | 0,02183 | −4,7% |
| CG traseiro, compensada | 0,04° | +0,00° | 0,02023 | 0,02183 | −7,3% |

Os C_D no ponto de projeto reproduzem exatamente os do item 3 (0,02081 e
0,02023), o que serve de verificação cruzada entre os dois scripts. A diferença
para o designTool está toda no arrasto induzido e de onda — o C_D0 é o mesmo por
construção (campo `# CDp`), e o AVL, sendo potencial linear, não modela onda.

Ajuste da Tab. 7 do enunciado (polar não trimada de CG traseiro, caso 5.b):
**C_D0 = 0,02010, C_Dα = 0,18763/rad, C_Dα² = 1,48578/rad², R² = 0,99999**.
Como α é o ângulo da fuselagem e a asa está a 4,5°, α = 0 já é praticamente o
ponto de projeto: **esse C_D0 não é o arrasto parasita**, e os três coeficientes
só fazem sentido juntos.

## Scripts dos itens 4 e 5

| Arquivo | Função |
| --- | --- |
| `avl_tools.py` | driver do `avl337.exe` em batch; leitores de `ft`, `fs`, `st`, `sb` |
| `ponto_projeto.py` | ponto de projeto (o mesmo do item 3), `it` lido de `resultados_q3/`, polar do designTool e condição de baixa velocidade |
| `xfoil_perfil.py` | `cl_max(Re)` do perfil do Lab 03, nos planos da corrente livre e normal ao bordo de ataque |
| `q4_secao_critica.py` | item 4 |
| `q5_polares.py` | item 5 |
| `q4_washout.py` | estudo de torção geométrica; alimenta as seções finais dos itens 4 e 5 |
| `q4_secao_critica.tex`, `q5_polares.tex`, `main.tex`, `compila.py` | relatório (tabelas e figuras entram por `\input`; os números do texto vêm de `tex_q*/macros.tex`, gerado pelos scripts) |

Ordem de execução:

```bash
python ponto_projeto.py     # resultados_comum/ponto_projeto.json
python xfoil_perfil.py      # resultados_comum/clmax_perfil_Re*.json  (~10 min)
python q4_secao_critica.py
python q5_polares.py
python q4_washout.py        # depende de resultados_q5/ (~25 min)
python compila.py           # relatorio.pdf
```

Os três últimos scripts de análise aceitam `--tex`, que refaz só CSV, tabelas e
figuras a partir do JSON já gravado, sem chamar o AVL. A pasta de trabalho `_avl_tmp/` leva o PID no
nome, então duas execuções simultâneas não se atrapalham.

`AVL_package/estudo_quebra_asa/scripts/_write_avl_trap.py` (o gerador dos
`fwd.avl`/`aft.avl`) foi atualizado para emitir o `ANGLE 4.5` e o `DESIGN it`,
que antes só existiam nos arquivos editados à mão e se perderiam numa
regeração.

## O que ainda falta da Tarefa 04

Itens 1 (texto de apresentação dos `.avl`), 2, 6, 7 e 8, e a Seção 3 (derivadas
de estabilidade). Os dados dos itens 6 e 7 já saem em
`resultados_q5/polar_*.csv` (colunas `alpha_deg` e `delta_e_deg`), e
`avl_tools.read_derivs` já lê as saídas `st`/`sb` da Seção 3.
