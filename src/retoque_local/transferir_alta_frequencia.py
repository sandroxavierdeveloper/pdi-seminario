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
         [--destino X0 X1 Y0 Y1] [--origem X0 X1 Y0]
  --destino  região cujo detalhe fino é trocado (padrão 300 845 506 622)
  --origem   janela de onde vem o detalhe: colunas X0:X1 e linha inicial Y0;
             a altura é a do destino (padrão 45 300 530)
Destino e origem precisam caber na imagem.
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter

from _comum import abrir_imagem, erro, validar_roi


def transferir(im, destino, origem):
    """im: imagem PIL RGB. destino=(X0, X1, Y0, Y1); origem=(OX0, OX1, OY0). Devolve array float32."""
    X0, X1, Y0, Y1 = destino
    ox0, ox1, oy0 = origem
    a = np.asarray(im).astype(np.float32)
    bl = np.asarray(im.filter(ImageFilter.GaussianBlur(5))).astype(np.float32)   # baixas frequências
    H, W = Y1 - Y0, X1 - X0
    # As baixas frequências do destino são mantidas; o detalhe vem de outra região.
    det_src = (a - bl)[oy0:oy0 + H, ox0:ox1]                     # altas frequências da origem
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
    return a


def main(argv=None):
    ap = argparse.ArgumentParser(description='Troca só o detalhe fino (altas frequências) de uma região pelo de outra.')
    ap.add_argument('entrada')
    ap.add_argument('saida')
    ap.add_argument('--destino', nargs=4, type=int, metavar=('X0', 'X1', 'Y0', 'Y1'), default=[300, 845, 506, 622])
    ap.add_argument('--origem', nargs=3, type=int, metavar=('X0', 'X1', 'Y0'), default=[45, 300, 530])
    a = ap.parse_args(argv)
    im = abrir_imagem(a.entrada)
    W, H = im.size
    validar_roi(*a.destino, W, H, nome='--destino')
    altura = a.destino[3] - a.destino[2]
    validar_roi(a.origem[0], a.origem[1], a.origem[2], a.origem[2] + altura, W, H,
                nome=f'--origem (altura {altura} px, igual à do destino)')
    r = transferir(im, a.destino, a.origem)
    Image.fromarray(r.clip(0, 255).astype(np.uint8)).save(a.saida, quality=95)
    print(f'{a.saida} gravado (destino {a.destino[0]}:{a.destino[1]}, {a.destino[2]}:{a.destino[3]})')


if __name__ == '__main__':
    main()
