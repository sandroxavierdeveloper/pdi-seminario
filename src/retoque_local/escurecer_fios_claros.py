#!/usr/bin/env python3
"""
escurecer_fios_claros.py: escurece só os pixels claros e pouco saturados
(fios grisalhos) dentro de uma elipse esfumada. Sem IA.

Técnicas (unidade da disciplina): máscara elíptica suavizada por filtro
gaussiano (2); luminância e saturação por pixel (3, 7); transformação de
intensidade (escala da luminância) com ponderação por máscara (2).

Uso: escurecer_fios_claros.py <entrada> <saida> <cx> <cy> <rx> <ry>
     (centro e raios da elipse, em pixels). A elipse precisa estar ao menos
     em parte dentro da imagem. Exemplo sintético: 200 120 150 90 (cabelo_sintetico.png).
"""
import argparse
import sys

import numpy as np
from PIL import Image, ImageFilter

from _comum import abrir_imagem, erro


def escurecer(a, cx, cy, rx, ry):
    """a: array RGB float32 (H, W, 3). Devolve a imagem com os fios claros escurecidos."""
    H, W, _ = a.shape
    yy, xx = np.mgrid[0:H, 0:W]
    el = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(np.float32)
    el = np.asarray(Image.fromarray((el * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(np.float32) / 255
    lum = a @ np.array([.299, .587, .114], np.float32)            # luminância
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)                            # saturação
    mask_el = el > .9
    escuro = np.percentile(lum[mask_el], 20) if np.any(mask_el) else np.percentile(lum[el > 0.1], 20) if np.any(el > 0.1) else np.percentile(lum, 20)  # tom da região escura (trata elipse pequena)
    w = el * np.clip((0.42 - sat) / 0.15, 0, 1) * np.clip((lum - escuro) / 25, 0, 1)   # só pixel claro e cinza
    alvo = escuro + (lum - escuro) * 0.3
    k = np.where(lum > 1, alvo / np.maximum(lum, 1), 1)
    return a * (1 - w[..., None]) + a * k[..., None] * w[..., None], w


def main(argv=None):
    ap = argparse.ArgumentParser(description='Escurece fios claros e pouco saturados dentro de uma elipse esfumada.')
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('cx', type=float, help='centro da elipse em x (px)')
    ap.add_argument('cy', type=float, help='centro da elipse em y (px)')
    ap.add_argument('rx', type=float, help='raio em x (px)')
    ap.add_argument('ry', type=float, help='raio em y (px)')
    a = ap.parse_args(argv)
    if a.rx <= 0 or a.ry <= 0:
        erro('os raios rx e ry precisam ser positivos.')
    img = np.asarray(abrir_imagem(a.entrada)).astype(np.float32)
    H, W, _ = img.shape
    if not (-a.rx < a.cx < W + a.rx and -a.ry < a.cy < H + a.ry):
        erro(f'a elipse (centro {a.cx:g},{a.cy:g}; raios {a.rx:g},{a.ry:g}) não toca a imagem de {W}x{H} px.')
    c, w = escurecer(img, a.cx, a.cy, a.rx, a.ry)
    Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(a.saida, quality=95)
    print(f'{a.saida}: {int((w > 0.5).sum())} pixels fortemente escurecidos')


if __name__ == '__main__':
    main()
