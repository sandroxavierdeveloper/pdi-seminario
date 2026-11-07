#!/usr/bin/env python3
"""
retoque_pele_roi.py: limiarização por cor em RGB dentro de uma ROI retangular
e transformação de intensidade (escala linear da luminância) só nos pixels
selecionados. Preserva o sombreado.

Caso de uso: um modelo generativo de imagem errou duas vezes ao pintar um
detalhe; a correção pontual foi feita por PDI clássico.

Técnicas (unidade da disciplina): limiarização por regra de cor em RGB (6, 7);
região de interesse retangular (6); luminância L = 0,3 R + 0,59 G + 0,11 B (3);
transformação de intensidade L' = a + b·L aplicada por pixel (2, 3).

Regra de seleção (tom de pele em RGB):  R > B + 6  e  R > G + 4.
Os pixels selecionados são escurecidos por L' = 8 + 0,10·L, mantendo o
sombreado relativo. O ponto fixo dessa reta é L = 8,89: abaixo dele o pixel
clarearia, mas tons de pele selecionados pela regra nunca são tão escuros.

Uso: retoque_pele_roi.py <entrada> <saida> <x0> <x1> <y0> <y1> [--mascara mascara.png]
"""
import sys

import numpy as np
from PIL import Image


def retocar(a, x0, x1, y0, y1, limiar_b=6, limiar_g=4, a0=8.0, b0=0.10):
    """Devolve (imagem retocada uint8, máscara bool). a: array RGB uint8."""
    f = a.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    sel = np.zeros(r.shape, bool)
    regra = (r > b + limiar_b) & (r > g + limiar_g)           # limiarização por cor
    sel[y0:y1, x0:x1] = regra[y0:y1, x0:x1]                   # restrita à ROI
    L = 0.3 * r + 0.59 * g + 0.11 * b                         # luminância
    Ln = a0 + b0 * L                                          # transformação de intensidade (fórmula do slide 2)
    k = np.where(L > 1, Ln / np.maximum(L, 1), 1.0)
    out = f.copy()
    out[sel] = f[sel] * k[sel][:, None]                       # escala os três canais igualmente
    return np.clip(out, 0, 255).astype(np.uint8), sel


if __name__ == '__main__':
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    if len(args) < 6:
        print(__doc__)
        sys.exit(2)
    entrada, saida = args[0], args[1]
    x0, x1, y0, y1 = map(int, args[2:6])
    img = np.asarray(Image.open(entrada).convert('RGB'))
    res, sel = retocar(img, x0, x1, y0, y1)
    Image.fromarray(res).save(saida)
    if '--mascara' in sys.argv:
        Image.fromarray((sel * 255).astype(np.uint8)).save(sys.argv[sys.argv.index('--mascara') + 1])
    print(f'{int(sel.sum())} pixels alterados na ROI ({x0}:{x1}, {y0}:{y1})')
