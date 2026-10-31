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
| src/retoque\_local/remover\_galhos\_mediana.py | Remove estruturas finas e escuras sobre um fundo | Filtro de mediana, limiar suave, abertura e fechamento |
| src/retoque\_local/transferir\_alta\_frequencia.py | Troca só o detalhe fino de uma região | Separação passa-baixas e passa-altas com filtro gaussiano |
| src/retoque\_local/gama\_e\_tom.py | Gama, ganho por canal e gradiente de cor | Transformação de intensidade, mistura ponderada |
| exemplos/gerar\_exemplos.py | Gera imagens sintéticas para testar os scripts |  |


## Cobertura da ementa

Este trabalho não pretende cobrir todas as unidades. O recorte cobre bem as de sequências e operações discretas, formação de imagens, morfologia, segmentação e ROI, e cobre em parte cores e descritores. **Wavelet não é usada.** Fourier aparece só no `avaliar_animatic.py` (FFT 1D da série temporal de movimento); a separação em baixas e altas frequências das imagens é feita no domínio espacial, com filtro gaussiano.

## O que não é PDI neste repositório

- O preenchimento neural do limpar\_legendas.py (LaMa, opcional) e o mapa de profundidade do animatic\_pdi.py (Depth Anything V2, opcional) são redes neurais usadas como caixa-preta. A máscara que alimenta o LaMa e todo o resto do pipeline são PDI clássico. Sem --modelo, o animatic\_pdi.py usa uma rampa vertical no lugar do mapa neural.

- Os modelos de geração de imagem que produziram as imagens originais não fazem parte deste código.

## Como rodar

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1) gerar imagens sintéticas
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

Os scripts nunca alteram os arquivos de entrada.

## Imagens

Este repositório contém apenas imagens sintéticas geradas por exemplos/gerar\_exemplos.py. As imagens usadas na apresentação em sala não estão aqui.

## Referência

MARQUES FILHO, O.; VIEIRA NETO, H. *Processamento Digital de Imagens*. Rio de Janeiro: Brasport, 1999.

