# AVL_package — NJ-0502

Pacote AVL da disciplina (arquivos do professor) mais os modelos AVL da NJ-0502 e o estudo da **quebra da asa (Yehudi) para alojar o trem de pouso principal**.

**Resumo do estudo:** na asa trapezoidal do Lab 02, a roda do trem principal passa **0,60 m além do bordo de fuga** (ξ_mlg = 100% da corda local). Reotimizando a aeronave com a nacele livre, a restrição de water spray e a margem estática verificada no AVL, e com uma quebra no bordo de fuga interno, o trem vai para **70% da corda com W0 0,16% menor**. Todas as restrições do Lab 02 continuam atendidas. **65% da corda não é viável** com o motor atual: faltam empuxo e margem estática.

---

## Conteúdo da pasta

| Arquivo | O que é | Status |
|---|---|---|
| `avl337.exe` | AVL 3.37 (Windows) | professor |
| `docs/` | Tutorial e manual do AVL | professor |
| `b737simple.avl`, `b737mod.avl`, `test.avl` | Modelos de exemplo (737 simplificado, em pés) | professor |
| `a1.dat`, `sd7037.dat`, `fuseB737_nondim.dat` | Perfis e forma da fuselagem usados pelos modelos | professor |
| **`nj0502_otim_quebra.avl`** | **Modelo vigente da NJ-0502**: asa reotimizada com quebra, EH, EV, fuselagem e naceles, em metros | **vigente** |
| **`nj0502_otim_quebra_inputs.py`** | Novos valores de `my_airplane` (variáveis de projeto + `z_n`) correspondentes ao modelo vigente | **vigente** |
| **`nj0502_otim_quebra_planta.png`** | Vista em planta: Lab 02 × reotimizada com quebra | **vigente** |
| `nj0502_baseline.avl` | Asa trapezoidal do Lab 02 em AVL (referência de comparação) | referência |
| `nj0502_quebra.avl`, `nj0502_quebra_planta.png`, `nj0502_quebra_zoom.png` | 1ª rodada: quebra em y = 12 m com a nacele em x_n = 21 m | **superado** (ver abaixo) |
| `estudo_quebra_asa/` | Scripts e resultados que geram o modelo vigente | — |

**Por que a 1ª rodada foi superada:** ela adiantava a nacele para x_n = 21 m, o que leva `SM_aft` a 15,4% no designTool e **viola a restrição `SM_aft ≤ 10%` do Lab 02**. Ela também mantinha a área de flap pela convenção do designTool, que inclui o trecho dentro da fuselagem. Com a corda de raiz maior, parte dessa área ficava fictícia.

## Como rodar o AVL

Execute a partir desta pasta, porque os modelos leem `a1.dat` e `fuseB737_nondim.dat` por caminho relativo:

```
avl337.exe
LOAD nj0502_otim_quebra.avl
OPER
A A 2        (alpha = 2 graus)
X            (executa)
ST           (derivadas de estabilidade; mostra o ponto neutro Xnp)
```

Controles definidos: `flap`, `slat`, `aileron`, `elevator`, `rudder`.

---

## Estudo da quebra da asa

### Por que o trem estava no bordo de fuga

`x_mlg`, `y_mlg` e `z_lg` **já eram variáveis do Lab 02**. O trem ficou no bordo de fuga porque a restrição `gear_te` (ξ_mlg ≤ 1,00, em `lab02_opt/otimizacao_NJ0502/opt_common.py`) está **ativa** no ótimo, junto com `SM_aft_max`, `tipback`, `tailstrike`, `tank`, `wheelspan`, `CLv`, `vt_te` e `ht_te`. Para mudar a posição do trem, é preciso mudar a asa ou relaxar essas restrições. Mover só o trem não resolve.

### Referências históricas (dados já presentes no repositório)

| | Fokker 100 (`standard_airplane('fokker100')`) | 737 (`b737simple.avl`) | NJ-0502 Lab 02 | NJ-0502 novo |
|---|---|---|---|---|
| ξ_mlg (trem / corda local) | **0,676** | — | 1,00 | **0,70** |
| Quebra / (b/2) | — | 60,2% | — | 45,5% |
| Ganho de corda de raiz pela quebra | — | +21% | — | +52% |
| c_quebra / c_raiz | — | 0,43 | — | 0,26 |
| Bordo de fuga interno | — | 7,5° | — | 0° (vertical) |
| Flap móvel exposto / S_w | 0,169 | — | 0,211 | 0,203 |
| Área alagada / S_w | 0,564 | — | 0,661 | 0,634 |
| Folga flap → aileron | 6% b/2 | — | 2% b/2 | 2% b/2 |

- O alvo de ~65–70% da corda é realista: bate com o Fokker 100 e com a própria NJ-0502, que tem tanque de 20% a 60% da corda e articulação do flap em 68%, o que põe a longarina traseira em ~65%.
- A quebra da NJ-0502 é **mais agressiva que a do 737**. O enflechamento de BA de ~37,5° faz a corda cair rápido com o bordo de fuga vertical.
- O flap da NJ-0502 é **maior que o do Fokker 100**, e a restrição de pouso tem ~28% de folga. Há espaço para reduzir `c_flap_c_wing`/`b_flap_b_wing`, que não são variáveis da otimização.

### Método

Estende o SLSQP do Lab 02 **sem alterar os arquivos do Lab 02**: as variáveis e restrições são acrescentadas em tempo de execução, em `scripts/opt_nac.py`.

1. **Novas variáveis:** `x_n`, `y_n`. `z_n` acompanha a asa e mantém a distância vertical da linha de base.
2. **Novas restrições:**
   - `spray`: motor fora do cone de 22° do trem de nariz, o mesmo desenhado em `plots.py`.
   - `nac_inlet` / `nac_exit`: entrada ≥ 0,25·L_n à frente do BA; saída ≤ 0,25·L_n à frente do BA.
   - `nac_gnd`: nacele ≥ 0,5 m do solo.
   - `nac_mlg`: borda interna da nacele ≥ y_mlg + 1 m, para o motor ficar por fora do trem.
3. **Quebra:** o bordo de fuga interno é vertical e preserva S, enflechamento de BA, corda de ponta e envergadura da asa otimizada. A quebra é dimensionada para pôr o trem em ξ_alvo da corda local, em forma fechada: `crank()` em `opt_nac.py`. Restrições:
   - `crank_cb`: c_quebra ≥ 2·c_ponta, ou seja, painel externo afilado como no 737.
   - `crank_ybmax`: y_quebra ≤ 60% b/2.
   - `crank_ybmin`: y_quebra ≥ y_mlg + 1 m.
4. **Estabilidade pelo AVL** (`scripts/opt_avl.py`). O designTool **não enxerga o efeito desestabilizante da nacele** nem o **avanço do ponto neutro causado pela quebra**. Sem correção, o otimizador explora essas lacunas: adianta o motor e o SM "parece" 10%, mas no AVL dá 6,7%. Por isso:
   - dentro do SLSQP, a quebra corrige CG da asa, MAC, tipback, cargas no trem de nariz e o NP (0,85 × deslocamento do quarto de corda do MAC);
   - fora do SLSQP, o NP do modelo completo no AVL recalibra a diferença até convergir (< 1 cm).
5. **Continuação** em ξ_alvo = 0,95 → 0,65, partindo sempre da solução anterior.

### Resultados

Continuação com correção pelo AVL, sem crédito de combustível, `crank_cb` com c_quebra ≥ 2·c_ponta (`resultados/avl_opt_rows_cons_cb2.json`):

| ξ_alvo | W0 vs Lab 02 | S_w [m²] | AR | quebra / (b/2) | SM_aft (AVL) | |
|---|---|---|---|---|---|---|
| 1,00 (sem quebra) | −0,78% | 384,7 | 10,84 | — | 10,0% | viável; o trem continua no BF |
| 0,90 | −1,16% | 383,4 | 10,99 | 31% | 10,0% | viável |
| 0,80 | −1,28% | 382,9 | 11,00 | 41% | 7,8% | viável |
| **0,70** | **−0,16%** | **386,6** | **10,72** | **45%** | **10,0%** | **viável → modelo vigente** |
| 0,675 | +4,12% | 391,1 | 8,71 | 51% | 10,0% | viável, mas AR cai muito |
| 0,65 | +8,90% | 470,0 | 7,93 | 54% | 15,7% | **inviável** (empuxo, SM_aft_max) |

**Modelo vigente (ξ = 0,70):**

| | Lab 02 | Novo |
|---|---|---|
| S_w [m²] / AR / Λ c/4 [°] | 386,48 / 10,64 / 35,38 | 386,62 / 10,72 / 35,11 |
| x_r,w [m] | 18,554 | 19,538 |
| Nacele x_n / y_n / z_n [m] | 24,30 / 11,00 / −2,90 | 19,60 / 9,92 / −2,99 |
| Trem x_mlg / y_mlg / z_lg [m] | 32,226 / 6,95 / −5,673 | 31,777 / 6,95 / −5,752 |
| Corda de raiz [m] | 10,04 | 15,21 (BF vertical em x = 34,74 m) |
| Quebra y [m] / corda [m] | — | 14,64 / 4,00 |
| ξ_mlg / folga roda → BF [m] | 1,00 / −0,60 | 0,70 / +2,37 |
| MAC [m] | 6,92 | 8,63 |
| Xnp AVL [m] / SM_aft / SM_fwd | 31,60 / 12,9% / 28,1% | 31,10 / 10,0% / 16,7% |
| W0 [kgf] | 276 276 | 275 834 |
| Folga de water spray [m] | 7,53 | 9,55 |

**Flap no modelo vigente:** c_f/c = 0,32, da lateral da fuselagem até 66% b/2, o mesmo fim do Lab 02 e a mesma folga para o aileron. A área móvel cai 4%, mas o **ΔCL_max do flap sobe ~5%**, porque a articulação interna cai de 30,8° para 13,8° de enflechamento. Manter a área *exposta* constante exigiria levar o flap até 71% b/2, invadindo o aileron.

### Hipóteses e limitações

- O designTool **não modela a quebra** no peso nem no arrasto da asa, nem no CL_max limpo. Esses termos continuam os da asa trapezoidal.
- **Sem crédito de combustível na quebra**: o tanque e o CG do combustível continuam os da asa trapezoidal, porque numa asa Yehudi típica a extensão do bordo de fuga é estrutura de flap. Com crédito (`01_otimiza_com_avl.py --fuel`), o otimizador encolhe a asa até os limites (S = 320 m², AR = 12). É otimista demais; ver `resultados/avl_opt_rows_fuel.json`, gerado por uma versão anterior do script com crédito sempre ligado e malha de 30 vórtices.
- As restrições de nacele (`nac_*`) e os limites da quebra (`crank_*`) são **hipóteses do estudo**, não do roteiro do Lab.
- No AVL: perfil `a1.dat` em todas as seções, washout linear de −3°, fuselagem com a forma do 737 escalada e nacele como anel sustentador. O Xnp é independente de `Xref`.
- O CG corrigido pela quebra usa a regra do designTool para o CG da asa (x_MAC + 0,4·MAC).

### Como reproduzir

Requer Python com `numpy`, `scipy`, `matplotlib` e `pandas`, e **Windows** (por causa do `avl337.exe`). Rode dentro de `estudo_quebra_asa/scripts/`:

```
python 01_otimiza_com_avl.py --cb2   # otimização + continuação em xi (demorado) -> resultados/avl_opt_rows_cons_cb2.json
python 02_relatorio_final.py         # relatório do ponto xi = 0,70 e do flap  -> resultados/final_design.json
python 03_gera_avl_final.py          # gera nj0502_otim_quebra.avl e nj0502_otim_quebra_inputs.py
python 04_figura_planta.py           # gera nj0502_otim_quebra_planta.png
```

Sem `--cb2`, a continuação usa c_quebra ≥ c_ponta e gera `avl_opt_rows_cons.json`. Esse caso chega a ξ = 0,65, mas com o painel externo retangular (c_quebra = c_ponta), uma asa pouco realista.

| Script | Função |
|---|---|
| `opt_nac.py` | Estende `opt_common` (Lab 02): variáveis `x_n`/`y_n`, restrições de nacele/spray/quebra, `crank()` |
| `opt_avl.py` | Correções da quebra dentro do SLSQP e laço externo de calibração do NP pelo AVL |
| `avl_check.py` | Planforma com quebra, escrita do `.avl` completo e leitura do Xnp |
