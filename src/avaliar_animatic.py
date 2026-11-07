#!/usr/bin/env python3
"""
avaliar_animatic.py: avalia um vídeo (animatic ou vídeo gerado) só com PDI
clássico, sem IA. É a etapa de AVALIAÇÃO do pipeline: decide, por medição,
se o movimento é real, se é só câmera e se há trechos congelados.

O vídeo é tratado como sinal discreto f(x, y, t): uma sequência de imagens
em tons de cinza reduzidas para --largura pixels.

Medidas por par de quadros consecutivos (unidade da disciplina entre parênteses):
  1. diferença absoluta média |f(t+1) - f(t)|               (2: operações aritméticas)
  2. SSIM com janela gaussiana 11x11, sigma 1,5               (2, 4: filtro gaussiano)
  3. fluxo óptico denso (Farnebäck) e ajuste de um modelo global de
     semelhança (translação + rotação + escala) por RANSAC. O modelo global
     é o movimento de CÂMERA; o resíduo é o movimento LOCAL (as coisas que
     se mexem de verdade), medido como % da área do quadro cujo resíduo
     passa de 1 px (limiarização). Em animatic 2.5D, o modelo 2D não absorve
     a paralaxe de profundidade, gerando resíduo que expressa esse relevo. (3: transformações geométricas)
Medidas sobre a série temporal:
  4. quadros CONGELADOS: sequências em que a diferença fica abaixo do limiar
     (limiarização 1D + contagem de sequências).              (1, 6)
  5. espectro do movimento local: FFT 1D da série, frequência dominante.   (1, 4)

Saída: resumo no terminal, <saida>/medidas.csv e <saida>/grafico.png.

Uso: python3 avaliar_animatic.py <video> [--saida PASTA] [--largura 320]
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

LIMIAR_CONGELADO = 0.30   # diferença média (0-255) abaixo disto = quadro repetido
SEG_CONGELADO_MIN = 0.5   # duração mínima (s) para contar como trecho congelado
LIMIAR_PIXEL = 1.0        # resíduo de fluxo (px/quadro) acima disto = pixel em movimento local
LIMIAR_AREA = 0.2         # % da área em movimento abaixo disto = "as coisas não se mexem"


def ssim(a, b):
    """SSIM médio entre duas imagens em tons de cinza (janela gaussiana 11x11, sigma 1,5)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mua, mub = g(a), g(b)
    saa, sbb, sab = g(a * a) - mua ** 2, g(b * b) - mub ** 2, g(a * b) - mua * mub
    m = ((2 * mua * mub + c1) * (2 * sab + c2)) / ((mua ** 2 + mub ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def fluxo_global_e_local(prev, nxt):
    """Devolve (câmera, área): magnitude média do movimento global (px) e % da área com resíduo
    de movimento acima de LIMIAR_PIXEL. A média do resíduo no quadro inteiro diluiria um
    objeto pequeno; por isso a medida de movimento local é a ÁREA em movimento (limiarização)."""
    flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev.shape
    ys, xs = np.mgrid[4:h:8, 4:w:8]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    dst = pts + flow[ys.ravel(), xs.ravel()]
    M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if M is None:
        return 0.0, 100.0 * float((np.linalg.norm(flow, axis=2) > LIMIAR_PIXEL).mean())
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    modelo = np.stack([M[0, 0] * gx + M[0, 1] * gy + M[0, 2] - gx,
                       M[1, 0] * gx + M[1, 1] * gy + M[1, 2] - gy], 2)
    return float(np.linalg.norm(modelo, axis=2).mean()), 100.0 * float((np.linalg.norm(flow - modelo, axis=2) > LIMIAR_PIXEL).mean())


def ler_quadros(caminho, largura):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        sys.exit(f'não consegui abrir {caminho}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    quadros = []
    while True:
        ok, q = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
        esc = largura / g.shape[1]
        quadros.append(cv2.resize(g, (largura, int(g.shape[0] * esc)), interpolation=cv2.INTER_AREA))
    cap.release()
    return quadros, fps


def sequencias_congeladas(diff, fps):
    """Lista de (inicio, fim) em quadros onde diff < limiar por pelo menos SEG_CONGELADO_MIN."""
    minimo = int(SEG_CONGELADO_MIN * fps)
    abaixo = np.asarray(diff) < LIMIAR_CONGELADO
    seqs, i = [], 0
    while i < len(abaixo):
        if abaixo[i]:
            j = i
            while j < len(abaixo) and abaixo[j]:
                j += 1
            if j - i >= minimo:
                seqs.append((i, j))
            i = j
        else:
            i += 1
    return seqs


def freq_dominante(serie, fps):
    """Frequência (Hz) de maior energia na FFT da série sem a média, ignorando DC."""
    s = np.asarray(serie, float)
    if len(s) < 16:
        return None
    espectro = np.abs(np.fft.rfft((s - s.mean()) * np.hanning(len(s))))
    freqs = np.fft.rfftfreq(len(s), 1.0 / fps)
    espectro[0] = 0
    return float(freqs[int(np.argmax(espectro))])


def grafico(caminho, series, fps, congeladas, piso=1.0):
    W, H, M = 900, 130 * len(series) + 40, 50
    img = np.full((H, W, 3), 255, np.uint8)
    n = len(series[0][1])
    for k, (nome, v) in enumerate(series):
        y0 = 20 + k * 130
        vmax = max(piso, max(v) * 1.1)   # piso: o ruído não pode parecer movimento grande
        cv2.rectangle(img, (M, y0), (W - 10, y0 + 100), (200, 200, 200), 1)
        for a, b in congeladas:
            cv2.rectangle(img, (M + int((W - M - 10) * a / n), y0 + 1), (M + int((W - M - 10) * b / n), y0 + 99), (230, 215, 190), -1)
        pts = np.array([[M + int((W - M - 10) * i / max(1, n - 1)), y0 + 100 - int(100 * x / vmax)] for i, x in enumerate(v)], np.int32)
        cv2.polylines(img, [pts], False, (143, 165, 31), 2, cv2.LINE_AA)
        cv2.putText(img, f'{nome}  (max {max(v):.2f})', (M + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f'tempo: {n / fps:.1f} s   (faixa azul = trecho congelado)', (M, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(caminho, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--saida', default='./saidas')
    ap.add_argument('--largura', type=int, default=320)
    a = ap.parse_args()
    quadros, fps = ler_quadros(a.video, a.largura)
    if len(quadros) < 3:
        sys.exit('vídeo curto demais')
    diff, simm, cam, loc = [], [], [], []
    for p, q in zip(quadros[:-1], quadros[1:]):
        # Filtro gaussiano 5x5 antes da diferença absoluta: remove ruído gaussiano/grão (sigma~5.5)
        # permitindo que quadros congelados com grão caiam abaixo do limiar (0.30):
        pf = cv2.GaussianBlur(p, (5, 5), 1.5).astype(np.float32)
        qf = cv2.GaussianBlur(q, (5, 5), 1.5).astype(np.float32)
        diff.append(float(np.abs(qf - pf).mean()))
        simm.append(ssim(p, q))
        c, l = fluxo_global_e_local(p, q)
        cam.append(c)
        loc.append(l)
    cong = sequencias_congeladas(diff, fps)
    frac = sum(b - a_ for a_, b in cong) / len(diff)
    ml = float(np.mean(loc))                       # % médio da área em movimento local
    veredito = 'PARADO: as coisas não se mexem' if ml < LIMIAR_AREA else 'há movimento local'
    os.makedirs(a.saida, exist_ok=True)
    with open(os.path.join(a.saida, 'medidas.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['par', 'diff_abs', 'ssim', 'camera_px', 'local_area_pct'])
        for i in range(len(diff)):
            w.writerow([i, f'{diff[i]:.4f}', f'{simm[i]:.4f}', f'{cam[i]:.4f}', f'{loc[i]:.4f}'])
    grafico(os.path.join(a.saida, 'grafico.png'),
            [('diferenca entre quadros', diff), ('movimento de camera (px)', cam), ('area em movimento local (%)', loc)], fps, cong)
    print(f'{os.path.basename(a.video)} · {len(quadros)} quadros · {fps:.0f} fps · {len(quadros) / fps:.1f} s')
    print(f'  SSIM médio entre quadros : {np.mean(simm):.3f}')
    print(f'  movimento de câmera      : {np.mean(cam):.2f} px/quadro')
    print(f'  área em movimento local  : {ml:.2f}% do quadro  → {veredito}')
    print(f'  trechos congelados       : {len(cong)} ({100 * frac:.0f}% do vídeo)' + ''.join(f' [{x / fps:.1f}s-{y / fps:.1f}s]' for x, y in cong))
    f0 = freq_dominante(loc, fps)
    print(f'  frequência dominante do movimento local: ' + (f'{f0:.2f} Hz' if f0 is not None else 'vídeo curto demais'))
    print(f'  saídas em {a.saida}/ (medidas.csv, grafico.png)')


if __name__ == '__main__':
    main()
