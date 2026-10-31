#!/usr/bin/env python3
"""Gera imagens SINTÉTICAS (sem arte de terceiros) para testar os scripts de src/.

  pagina_sintetica.png : fundo colorido com texturas e dois balões de fala
                         (miolo claro, contorno escuro, letras) para limpar_legendas.py
  pes_sinteticos.png   : cena escura com duas formas de tom de pele para
                         retoque_pele_roi.py (ROI sugerida: 120 340 200 330)
  video_teste.mp4      : 4 s a 24 fps. Um disco se move nos 2 primeiros segundos e
                         os 2 últimos ficam CONGELADOS (para avaliar_animatic.py).
                         Não é versionado (o .gitignore ignora .mp4); gere-o aqui.

Uso: python3 exemplos/gerar_exemplos.py [pasta_de_saida]   (padrão: exemplos)
"""
import os
import sys

import cv2
import numpy as np

saida = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# ── Página com balões ────────────────────────────────────────────────────
H, W = 700, 1000
pag = np.zeros((H, W, 3), np.uint8)
for y in range(H):                                   # gradiente de fundo (verde → marrom)
    t = y / H
    pag[y, :] = (int(60 + 40 * t), int(110 - 40 * t), int(70 + 20 * t))
pag = np.clip(pag.astype(np.int16) + rng.normal(0, 12, pag.shape), 0, 255).astype(np.uint8)  # textura
for _ in range(60):                                  # "folhagem"
    cv2.circle(pag, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(8, 30)),
               (int(rng.integers(30, 90)), int(rng.integers(80, 150)), int(rng.integers(40, 90))), -1)
def balao(img, cx, cy, rx, ry, linhas):
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (15, 15, 15), 4)
    y = cy - 12 * (len(linhas) - 1)
    for t in linhas:
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(img, t, (cx - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
        y += 28
balao(pag, 260, 170, 200, 90, ['OLHA O CEU,', 'VAI CHOVER LOGO', 'NESTA TARDE'])
balao(pag, 720, 480, 210, 95, ['NAO TENHO MEDO', 'DE UMA CHUVA', 'DE VERAO'])
cv2.imwrite(os.path.join(saida, 'pagina_sintetica.png'), pag)

# ── Cena escura com formas "de pele" ─────────────────────────────────────
cena = np.full((480, 480, 3), (60, 70, 62), np.uint8)
cena = np.clip(cena.astype(np.int16) + rng.normal(0, 4, cena.shape), 0, 255).astype(np.uint8)
cv2.rectangle(cena, (0, 0), (480, 140), (30, 22, 20), -1)                  # "calça" escura (BGR)
cv2.ellipse(cena, (200, 250), (70, 24), 8, 0, 360, (140, 150, 205), -1)    # pele (BGR: R > G > B)
cv2.ellipse(cena, (260, 310), (80, 28), 4, 0, 360, (130, 140, 195), -1)
cv2.imwrite(os.path.join(saida, 'pes_sinteticos.png'), cena)
# ── Vídeo de teste: movimento local e depois congelamento ────────────────
fundo = cena.copy()
vw = cv2.VideoWriter(os.path.join(saida, 'video_teste.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 24, (480, 480))
for i in range(96):
    q = fundo.copy()
    k = min(i, 47)                                   # a partir do quadro 48 nada mais muda
    cv2.circle(q, (60 + int(7.5 * k), 400 - int(2.0 * k)), 28, (40, 180, 240), -1)
    vw.write(q)
vw.release()
print('gerado em', saida)
