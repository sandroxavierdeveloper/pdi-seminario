# PDI em pipelines híbridos de IA

Código de apoio do seminário **Aplicação de Técnicas Clássicas de PDI para Pré e Pós-Processamento em Pipelines Híbridos de IA**, da disciplina de Processamento Digital de Sinais e Imagens (UFG).

## Equipe PDI

| Nome completo | Matrícula | Curso |
| - | - | - |
| Bruno Barros Ribeiro | 202503994 | Bacharelado em Inteligência Artificial |
| Sandro Xavier de Faria | 202403925 | Bacharelado em Inteligência Artificial |

## Pipeline

1. Gravura de Gustave Doré (domínio público) como entrada.
2. Imagem nova inspirada na gravura, gerada por modelo de imagem (IA, fora deste código).
3. **Análise e correção de erros por PDI** (`limpar_legendas.py`, `retoque_local/`).
4. **Animatic** sem IA (`animatic_pdi.py`).
5. **Avaliação do animatic por PDI** (`avaliar_animatic.py`).
6. Se aprovado, vídeo final gerado por modelo de vídeo (IA, fora deste código).

As etapas 3 a 5 são o que este repositório implementa.

## Ideia

Modelos generativos de imagem erram detalhes que um algoritmo determinístico corrige: texto deformado, legendas que reaparecem, um detalhe pintado errado. Os scripts abaixo tratam a imagem como uma matriz de pixels (NumPy) e a modificam com técnicas clássicas de PDI, de forma reprodutível: mesma entrada, mesma saída.

## Conteúdo

| Arquivo | O que faz | Técnicas |
| - | - | - |
| src/limpar\_legendas.py | Detecta balões e legendas em páginas de quadrinhos e gera a máscara (e, opcionalmente, a imagem limpa) | Limiarização em HSV, morfologia, componentes conexos, descritores de região |
| src/animatic\_pdi.py | Anima uma imagem parada (zoom, deriva, paralaxe, névoa, grão) e grava um mp4 | Remapeamento geométrico, interpolação, filtro gaussiano, operações aritméticas |
| src/avaliar\_animatic.py | Avalia um vídeo: movimento de câmera e movimento local (fluxo óptico), SSIM, diferença entre quadros, trechos congelados e espectro do movimento | Fluxo óptico, ajuste global por RANSAC, limiarização, SSIM, FFT 1D |
| src/retoque\_local/retoque\_pele\_roi.py | Corrige um detalhe dentro de uma ROI por regra de cor e escala de luminância | Limiarização em RGB, ROI, transformação de intensidade |
| src/retoque\_local/escurecer\_fios\_claros.py | Escurece pixels claros e pouco saturados numa elipse esfumada | Máscara gaussiana, luminância e saturação |
| src/retoque\_local/transferir\_galhos\_mediana.py | Isola estruturas finas e escuras (galhos) numa imagem e as transfere para uma imagem base | Filtro de mediana, limiar suave, abertura e fechamento |
| src/retoque\_local/transferir\_alta\_frequencia.py | Troca só o detalhe fino de uma região | Separação passa-baixas e passa-altas com filtro gaussiano |
| src/retoque\_local/gama\_e\_tom.py | Gama, ganho por canal e gradiente de cor | Transformação de intensidade, mistura ponderada |
| exemplos/gerar\_exemplos.py | Gera imagens e vídeo sintéticos para testar os scripts |  |
| run\_all.sh | Gera os exemplos e roda toda a demonstração, gravando em saidas/ |  |
| tests/ | Testes automatizados (pytest) sobre os exemplos sintéticos |  |


## Cobertura da ementa

Este trabalho não pretende cobrir todas as unidades. O recorte cobre bem as de sequências e operações discretas, formação de imagens, morfologia, segmentação e ROI, e cobre em parte cores e descritores. **Wavelet não é usada.** Fourier aparece só no `avaliar_animatic.py` (FFT 1D da série temporal de movimento); a separação em baixas e altas frequências das imagens é feita no domínio espacial, com filtro gaussiano.

## O que não é PDI neste repositório

- O preenchimento neural do limpar\_legendas.py (LaMa, opcional) e o mapa de profundidade do animatic\_pdi.py (Depth Anything V2, opcional) são redes neurais usadas como caixa-preta. A máscara que alimenta o LaMa e todo o resto do pipeline são PDI clássico. Sem --modelo, o animatic\_pdi.py usa uma rampa vertical no lugar do mapa neural.

- Os modelos de geração de imagem que produziram as imagens originais não fazem parte deste código.

## Requisitos

- **Python 3.10 ou superior** (testado com 3.14).
- **ffmpeg obrigatório** no PATH para o `animatic_pdi.py` (grava o mp4); o script avisa se não o encontrar. Os demais scripts não o usam.
- Dependências Python em `requirements.txt` (NumPy, OpenCV, Pillow e pytest).

## Como rodar

**Antes de tudo, gere os exemplos:** a pasta `exemplos/` só traz duas imagens versionadas; as demais imagens sintéticas e o vídeo de teste (`video_teste.mp4`, ignorado pelo git) são criados por `exemplos/gerar_exemplos.py`. Os passos 2 a 5 dependem desse passo 1.

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) gerar imagens e vídeo sintéticos (obrigatório antes dos demais)
python3 exemplos/gerar_exemplos.py

# 2) detectar balões (só máscara, sem rede neural)
python3 src/limpar_legendas.py exemplos/pagina_sintetica.png --so-mascara --saida saidas

# 3) retoque por regra de cor dentro de uma ROI (x0 x1 y0 y1)
python3 src/retoque_local/retoque_pele_roi.py exemplos/pes_sinteticos.png saidas/pes.png 120 340 200 340 --mascara saidas/pes_mascara.png

# 4) animar uma imagem parada (precisa do ffmpeg)
python3 src/animatic_pdi.py exemplos/pagina_sintetica.png saidas/animatic.mp4 --segundos 3 --largura 640

# 5) avaliar um vídeo: câmera x movimento local, SSIM, congelamento
python3 src/avaliar_animatic.py exemplos/video_teste.mp4 --saida saidas/avaliacao
python3 src/avaliar_animatic.py saidas/animatic.mp4 --saida saidas/avaliacao_animatic
```

A ROI do passo 3 (`120 340 200 340`) é a mesma que o `gerar_exemplos.py` documenta: cobre as duas formas de tom de pele de `pes_sinteticos.png` (480x480). Os scripts de retoque conferem se a ROI cabe na imagem e, se não couber, encerram com uma mensagem de erro.

### Demonstração completa

```
./run_all.sh
```

Gera os exemplos e roda todos os scripts (inclusive os quatro retoques restantes), gravando tudo em `saidas/`. Requer ffmpeg.

### Opções de linha de comando

- `animatic_pdi.py --direcao "texto"`: descrição do plano, em inglês, de onde o script tira a intenção de câmera e o tipo de plano. Palavras como `push-in`, `pull back`, `pan to the left/right` definem zoom e deriva; `close-up`/`portrait` escolhem o tipo *rosto* (sem paralaxe); `torch`/`flame`/`fire` escolhem *fogo* (tremor de luz quente); o resto é *paisagem* (paralaxe e névoa). Sem `--direcao`, a câmera faz um zoom suave de 1,0 a 1,06. `--tipo auto|paisagem|fogo|rosto` força o tipo. Exemplo: `--direcao "slow push-in on a torch-lit corridor"`.
- `limpar_legendas.py`: `--so-mascara` (só detecta) e `--saida PASTA` (padrão `./saidas`).
- Retoques (`src/retoque_local/`): ROIs e coordenadas são argumentos, com os valores originais como padrão quando existiam (`--roi`/`--excluir` em `transferir_galhos_mediana.py`; `--destino`/`--origem` em `transferir_alta_frequencia.py`). Use `-h` em cada script.

### Limiares padrão do `avaliar_animatic.py`

Definidos no topo do código:

| Constante | Valor | Significado |
| - | - | - |
| `LIMIAR_CONGELADO` | 0,30 | diferença média absoluta (escala 0 a 255, após suavização gaussiana 5x5) abaixo da qual o par de quadros conta como repetido |
| `SEG_CONGELADO_MIN` | 0,5 s | duração mínima de uma sequência de quadros repetidos para virar "trecho congelado" |
| `LIMIAR_PIXEL` | 1,0 px | resíduo de fluxo óptico (depois de descontar o movimento global de câmera) acima do qual o pixel conta como em movimento local |
| `LIMIAR_AREA` | 0,2 % | se a área média em movimento local fica abaixo disso, o veredito é "PARADO: as coisas não se mexem" |

Outros padrões: `--largura 320` (largura de trabalho dos quadros) e `--saida ./saidas`. No `video_teste.mp4` (4 s, metade parada) o script acusa cerca de 50% de vídeo congelado.

## Testes

```
python3 -m pytest
```

Os testes (`tests/test_pipeline.py`) geram os exemplos sintéticos e verificam: 2 balões detectados em `pagina_sintetica.png`; cerca de 50% de congelamento em `video_teste.mp4`; os cinco retoques rodando nas imagens sintéticas; e o aviso de erro quando a ROI não cabe na imagem. Não precisam de ffmpeg (que só o animatic e o `run_all.sh` usam) nem de rede.

## Imagens

Este repositório contém apenas imagens sintéticas geradas por exemplos/gerar\_exemplos.py. As imagens usadas na apresentação em sala não estão aqui.

## Referência

MARQUES FILHO, O.; VIEIRA NETO, H. *Processamento Digital de Imagens*. Rio de Janeiro: Brasport, 1999.

