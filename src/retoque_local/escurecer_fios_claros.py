#!/usr/bin/env python3
"""
escurecer_fios_claros.py: escurece só os pixels claros e pouco saturados
(fios grisalhos) dentro de uma elipse esfumada. Sem IA.

Técnicas (unidade da disciplina): máscara elíptica suavizada por filtro
gaussiano (2); luminância e saturação por pixel (3, 7); transformação de
intensidade (escala da luminância) com ponderação por máscara (2).

Uso: escurecer_fios_claros.py <entrada> <saida> <cx> <cy> <rx> <ry>
     (centro e raios da elipse, em pixels)
"""
import sys

import numpy as np
from PIL import Image, ImageFilter

src, out, cx, cy, rx, ry = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:7])
I = Image.open(src).convert('RGB')
a = np.asarray(I).astype(np.float32)
H, W, _ = a.shape
yy, xx = np.mgrid[0:H, 0:W]
el = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(np.float32)
el = np.asarray(Image.fromarray((el * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(np.float32) / 255
lum = a @ np.array([.299, .587, .114], np.float32)            # luminância
mx, mn = a.max(2), a.min(2)
sat = (mx - mn) / np.maximum(mx, 1)                            # saturação
escuro = np.percentile(lum[el > .9], 20)                       # tom da região escura
w = el * np.clip((0.42 - sat) / 0.15, 0, 1) * np.clip((lum - escuro) / 25, 0, 1)   # só pixel claro e cinza
alvo = escuro + (lum - escuro) * 0.3
k = np.where(lum > 1, alvo / np.maximum(lum, 1), 1)
c = a * (1 - w[..., None]) + a * k[..., None] * w[..., None]
Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(out, quality=95)
