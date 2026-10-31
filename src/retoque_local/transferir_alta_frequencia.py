#!/usr/bin/env python3
"""
transferir_alta_frequencia.py: separa a imagem em baixas e altas frequências
com um filtro gaussiano e troca só o detalhe fino (altas frequências) de uma
região pelo de outra região da mesma imagem. Sem IA.

Técnicas (unidade da disciplina): filtro passa-baixas gaussiano (2, 4);
imagem de detalhe = original menos suavizada, ou seja, um passa-altas (4);
limiarização suave por luminância (6); máscara com bordas em rampa (2).

É o exemplo do projeto que melhor ilustra a ideia da unidade de frequência
sem usar a transformada de Fourier: low = G(a), high = a - G(a).

Uso: transferir_alta_frequencia.py <entrada> <saida>
Caixa de destino (X0, X1, Y0, Y1) e janela de origem do detalhe ficam no código.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

src, out = sys.argv[1:3]
im = Image.open(src).convert('RGB')
a = np.asarray(im).astype(np.float32)
bl = np.asarray(im.filter(ImageFilter.GaussianBlur(5))).astype(np.float32)   # baixas frequências
X0, X1, Y0, Y1 = 300, 845, 506, 622                          # região de destino
H, W = Y1 - Y0, X1 - X0
# As baixas frequências do destino são mantidas; o detalhe vem de outra região.
det_src = (a - bl)[530:530 + H, 45:300]                      # altas frequências da origem
pw = det_src.shape[1]
reps = W // pw + 2
det = np.concatenate([det_src] * reps, 1)[:, :W]
low = bl[Y0:Y1, X0:X1]
reg = a[Y0:Y1, X0:X1]
lum = low.mean(2, keepdims=True)
m = np.clip((lum - 70) / 40, 0, 1)                           # ignora a parte escura
wy = np.clip(np.minimum(np.arange(H), H - 1 - np.arange(H)) / 12, 0, 1)[:, None, None]
wx = np.clip(np.minimum(np.arange(W), W - 1 - np.arange(W)) / 20, 0, 1)[None, :, None]
m = m * wy * wx
a[Y0:Y1, X0:X1] = reg * (1 - m) + (low + det * 1.1) * m
Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(out, quality=95)
