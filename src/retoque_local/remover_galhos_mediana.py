#!/usr/bin/env python3
"""
remover_galhos_mediana.py: remove estruturas finas e escuras sobre o céu
(galhos) e devolve o fundo da imagem base. Sem IA.

Técnicas (unidade da disciplina): filtro de mediana 31x31 estima o fundo sem
os detalhes finos (2, 4); diferença de luminância entre imagem e fundo (2);
limiarização suave por rampa (6); limiarização em saturação HSV (7);
abertura/fechamento morfológico com filtros de máximo e mínimo (5);
suavização gaussiana da máscara e mistura ponderada (2).

Uso: remover_galhos_mediana.py <imagem_com_galhos> <imagem_base> <saida> <mascara_saida>
Parâmetros de ROI (X0, X1, Y0, Y1) e a exclusão da base ficam no código.
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

v1, base, out, mout = sys.argv[1:5]
I = Image.open(v1).convert('RGB')
a = np.asarray(I).astype(np.float32)
b = np.asarray(Image.open(base).convert('RGB')).astype(np.float32)
X0, X1, Y0, Y1 = 140, 480, 170, 400                        # ROI retangular
lum = lambda x: x @ np.array([.299, .587, .114], np.float32)
bg = np.asarray(I.filter(ImageFilter.MedianFilter(31))).astype(np.float32)   # fundo sem os detalhes finos
m = np.zeros(a.shape[:2], np.float32)
dd = lum(bg) - lum(a)                                      # quanto o pixel é mais escuro que o fundo
m[Y0:Y1, X0:X1] = np.clip((dd[Y0:Y1, X0:X1] - 6) / 18, 0, 1)
hsv = np.asarray(I.convert('HSV')).astype(np.float32)
m *= np.clip((hsv[..., 1] - 35) / 30, 0, 1)                # sem saturação (risco cinza) não conta
m[368:, :235] = 0                                          # exclui uma região da base, à esquerda

# Máximo 9x9 (dilatação) seguido de mínimo 7x7 (erosão) fecha buracos da máscara.
M = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(7)).filter(ImageFilter.GaussianBlur(1.2))
m = np.asarray(M).astype(np.float32) / 255
pl = np.asarray(I.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
c = b * (1 - m[..., None]) + pl * m[..., None]
Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(out, quality=95)
M.save(mout)
