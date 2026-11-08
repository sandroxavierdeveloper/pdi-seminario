#!/usr/bin/env python3
"""
transferir_galhos_mediana.py: isola as estruturas finas e escuras sobre o céu
(galhos/planta) da imagem v1 e as transfere para a imagem base, que mantém o
próprio fundo em todo o resto. Sem IA. O resultado é a base COM a planta da v1:
c = base·(1−m) + v1·m. (Antes chamado remover_galhos_mediana.py.)

Técnicas (unidade da disciplina): filtro de mediana 31x31 estima o fundo sem
os detalhes finos (2, 4); diferença de luminância entre imagem e fundo (2);
limiarização suave por rampa (6); limiarização em saturação HSV (7);
abertura/fechamento morfológico com filtros de máximo e mínimo (5);
suavização gaussiana da máscara e mistura ponderada (2).

Uso: transferir_galhos_mediana.py <imagem_com_galhos> <imagem_base> <saida> <mascara_saida>
         [--roi X0 X1 Y0 Y1] [--excluir X0 X1 Y0 Y1]
  --roi      retângulo onde procurar os galhos (padrão 140 480 170 400)
  --excluir  retângulo da máscara que é zerado, mantendo a base (padrão 0 235 368 <altura>)
As duas imagens precisam ter o mesmo tamanho.
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter

from _comum import abrir_imagem, erro, validar_roi


def transferir(I, base, roi, excluir):
    """I, base: imagens PIL RGB do mesmo tamanho. Devolve (resultado float32, máscara PIL 'L')."""
    X0, X1, Y0, Y1 = roi
    a = np.asarray(I).astype(np.float32)
    b = np.asarray(base).astype(np.float32)
    lum = lambda x: x @ np.array([.299, .587, .114], np.float32)
    bg = np.asarray(I.filter(ImageFilter.MedianFilter(31))).astype(np.float32)   # fundo sem os detalhes finos
    m = np.zeros(a.shape[:2], np.float32)
    dd = lum(bg) - lum(a)                                      # quanto o pixel é mais escuro que o fundo
    m[Y0:Y1, X0:X1] = np.clip((dd[Y0:Y1, X0:X1] - 6) / 18, 0, 1)
    hsv = np.asarray(I.convert('HSV')).astype(np.float32)
    m *= np.clip((hsv[..., 1] - 35) / 30, 0, 1)                # sem saturação (risco cinza) não conta
    ex0, ex1, ey0, ey1 = excluir
    m[ey0:ey1, ex0:ex1] = 0                                    # exclui uma região, mantendo a base

    # Máximo 9x9 (dilatação) seguido de mínimo 7x7 (erosão) fecha buracos da máscara.
    M = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(7)).filter(ImageFilter.GaussianBlur(1.2))
    m = np.asarray(M).astype(np.float32) / 255
    pl = np.asarray(I.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)   # planta da v1, levemente suavizada
    return b * (1 - m[..., None]) + pl * m[..., None], M


def main(argv=None):
    ap = argparse.ArgumentParser(description='Transfere galhos finos e escuros da imagem v1 para a imagem base.')
    ap.add_argument('imagem_com_galhos')
    ap.add_argument('imagem_base')
    ap.add_argument('saida')
    ap.add_argument('mascara_saida')
    ap.add_argument('--roi', nargs=4, type=int, metavar=('X0', 'X1', 'Y0', 'Y1'), default=[140, 480, 170, 400])
    ap.add_argument('--excluir', nargs=4, type=int, metavar=('X0', 'X1', 'Y0', 'Y1'), default=[0, 235, 368, 10 ** 9],
                    help='região da máscara zerada (será cortada nos limites da imagem)')
    a = ap.parse_args(argv)
    I = abrir_imagem(a.imagem_com_galhos)
    base = abrir_imagem(a.imagem_base)
    if I.size != base.size:
        erro(f'as imagens têm tamanhos diferentes: {I.size[0]}x{I.size[1]} e {base.size[0]}x{base.size[1]}.')
    W, H = I.size
    validar_roi(*a.roi, W, H, nome='--roi')
    ex = a.excluir
    ex = [max(0, ex[0]), min(W, ex[1]), max(0, ex[2]), min(H, ex[3])]
    c, M = transferir(I, base, a.roi, ex)
    Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).save(a.saida, quality=95)
    M.save(a.mascara_saida)
    print(f'{a.saida} e {a.mascara_saida} gravados (ROI {a.roi[0]}:{a.roi[1]}, {a.roi[2]}:{a.roi[3]})')


if __name__ == '__main__':
    main()
