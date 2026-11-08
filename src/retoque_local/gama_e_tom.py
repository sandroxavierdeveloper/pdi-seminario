#!/usr/bin/env python3
"""
gama_e_tom.py: correção de gama, ganho por canal e um gradiente vertical de
cor no céu, para dar um tom de entardecer. Sem IA.

Técnicas (unidade da disciplina): transformação de intensidade por potência
(correção de gama) (3); ganho por canal RGB (7); mistura ponderada por uma
máscara vertical (2).

Uso: gama_e_tom.py <entrada> <saida> [--gama 1.18]
"""
import argparse

import numpy as np
from PIL import Image

from _comum import abrir_imagem


def tom(a, gama=1.18):
    """a: array RGB float32 em [0, 1]. Devolve a imagem com gama, ganho e gradiente aplicados."""
    H = a.shape[0]
    a = a ** gama                                  # gama: escurece os meios-tons
    a = a * np.array([1.04, 0.90, 0.74])           # ganho por canal: aquece a imagem
    y = np.linspace(0, 1, H)[:, None, None]
    sky = np.clip(1 - y / 0.55, 0, 1)              # peso do topo da imagem
    a = a * (1 - 0.35 * sky) + 0.35 * sky * a * np.array([0.95, 0.80, 0.92])
    hz = np.exp(-((y - 0.57) / 0.06) ** 2)         # brilho gaussiano no horizonte
    return a + hz * np.array([0.10, 0.05, 0.0])


def main(argv=None):
    ap = argparse.ArgumentParser(description='Gama, ganho por canal e gradiente de cor (tom de entardecer).')
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('--gama', type=float, default=1.18, help='expoente da correção de gama (padrão 1.18)')
    a = ap.parse_args(argv)
    img = np.asarray(abrir_imagem(a.entrada)).astype(np.float32) / 255
    r = tom(img, a.gama)
    Image.fromarray((r.clip(0, 1) * 255).astype(np.uint8)).save(a.saida, quality=95)
    print(f'{a.saida} gravado')


if __name__ == '__main__':
    main()
