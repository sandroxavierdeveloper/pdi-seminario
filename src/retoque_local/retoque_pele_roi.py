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
import argparse

import numpy as np
from PIL import Image

from _comum import abrir_imagem, validar_roi


def retocar(a, x0, x1, y0, y1, limiar_b=6, limiar_g=4, a0=8.0, b0=0.10):
    """Devolve (imagem retocada uint8, máscara bool). a: array RGB uint8."""
    f = a.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    sel = np.zeros(r.shape, bool)
    regra = (r > b + limiar_b) & (r > g + limiar_g)           # limiarização por cor
    sel[y0:y1, x0:x1] = regra[y0:y1, x0:x1]                   # restrita à ROI
    L = 0.3 * r + 0.59 * g + 0.11 * b                         # luminância
    Ln = a0 + b0 * L                                          # transformação de intensidade
    k = np.where(L > 1, Ln / np.maximum(L, 1), 1.0)
    out = f.copy()
    out[sel] = f[sel] * k[sel][:, None]                       # escala os três canais igualmente
    return np.clip(out, 0, 255).astype(np.uint8), sel


def main(argv=None):
    ap = argparse.ArgumentParser(description='Escurece pixels de tom de pele dentro de uma ROI retangular.')
    ap.add_argument('entrada')
    ap.add_argument('saida')
    for n in ('x0', 'x1', 'y0', 'y1'):
        ap.add_argument(n, type=int)
    ap.add_argument('--mascara', help='grava a máscara dos pixels alterados neste arquivo')
    a = ap.parse_args(argv)
    img = np.asarray(abrir_imagem(a.entrada))
    validar_roi(a.x0, a.x1, a.y0, a.y1, img.shape[1], img.shape[0])
    res, sel = retocar(img, a.x0, a.x1, a.y0, a.y1)
    Image.fromarray(res).save(a.saida)
    if a.mascara:
        Image.fromarray((sel * 255).astype(np.uint8)).save(a.mascara)
    print(f'{int(sel.sum())} pixels alterados na ROI ({a.x0}:{a.x1}, {a.y0}:{a.y1})')


if __name__ == '__main__':
    main()
