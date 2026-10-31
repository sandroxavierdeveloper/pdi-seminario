#!/usr/bin/env python3
"""
gama_e_tom.py: correção de gama, ganho por canal e um gradiente vertical de
cor no céu, para dar um tom de entardecer. Sem IA.

Técnicas (unidade da disciplina): transformação de intensidade por potência
(correção de gama) (3); ganho por canal RGB (7); mistura ponderada por uma
máscara vertical (2).

Uso: gama_e_tom.py <entrada> <saida>
"""
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1:3]
a = np.asarray(Image.open(src).convert('RGB')).astype(np.float32) / 255
H = a.shape[0]
a = a ** 1.18                                  # gama: escurece os meios-tons
a *= np.array([1.04, 0.90, 0.74])              # ganho por canal: aquece a imagem
y = np.linspace(0, 1, H)[:, None, None]
sky = np.clip(1 - y / 0.55, 0, 1)              # peso do topo da imagem
a = a * (1 - 0.35 * sky) + 0.35 * sky * a * np.array([0.95, 0.80, 0.92])
hz = np.exp(-((y - 0.57) / 0.06) ** 2)         # brilho gaussiano no horizonte
a += hz * np.array([0.10, 0.05, 0.0])
Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8)).save(out, quality=95)
