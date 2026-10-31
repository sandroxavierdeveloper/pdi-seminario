#!/usr/bin/env python3
"""
animatic_pdi.py: dá movimento a uma imagem parada usando só técnicas
determinísticas de PDI, e grava um vídeo mp4 (precisa do ffmpeg no PATH).

Técnicas usadas (unidades da disciplina entre parênteses):
  · Transformação geométrica por remapeamento (cv2.remap): zoom, deriva
    horizontal e tremor de câmera como deslocamento por pixel.        (3)
  · Paralaxe 2.5D: o deslocamento de cada pixel é proporcional a um mapa
    de profundidade, então o que está perto se move mais que o fundo.
  · Reamostragem com interpolação (resize cúbico e Lanczos).          (3)
  · Filtragem espacial gaussiana: suaviza o mapa de profundidade, a
    máscara de luz quente e o campo de névoa.                         (2, 4)
  · Limiarização em HSV: isola a luz quente (tom laranja, claro).     (6, 7)
  · Operações aritméticas por pixel: vinheta, mistura de névoa e grão
    (ruído gaussiano aditivo com semente fixa).                       (2)
  · Séries discretas suaves (soma de senos): tremor reprodutível.     (1, 2)

Mapa de profundidade:
  · Com --modelo, usa uma rede neural (Depth Anything V2 small, ONNX) como
    caixa-preta. O arquivo do modelo NÃO acompanha este repositório.
  · Sem --modelo, usa uma rampa vertical (topo longe, base perto), que é
    PDI puro e permite rodar o script em qualquer máquina.

Uso:
  python3 animatic_pdi.py <imagem> <saida.mp4> [--segundos 8] [--direcao "texto"]
                          [--tipo auto|paisagem|fogo|rosto] [--largura 1920]
                          [--modelo caminho/depth_anything_v2_small.onnx]
"""
import argparse
import os
import re
import subprocess
import sys

import cv2
import numpy as np

FPS = 24


def profundidade(img: np.ndarray, modelo: str | None) -> np.ndarray:
    """Profundidade RELATIVA 0..1 (1 = perto), suavizada para o remapeamento não rasgar contornos."""
    h, w = img.shape[:2]
    if modelo:
        import onnxruntime as ort
        s = ort.InferenceSession(modelo, providers=['CPUExecutionProvider'])
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
        d = s.run(None, {s.get_inputs()[0].name: x})[0].squeeze()
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
        d = (d - d.min()) / max(1e-6, d.max() - d.min())
    else:
        d = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None], (1, w))
    return cv2.GaussianBlur(d, (0, 0), sigmaX=w / 160).astype(np.float32)


def tipo_do_plano(direcao: str) -> str:
    """Pela descrição do plano: rosto de perto não leva paralaxe (estica feições);
    luz de chama leva tremor; plano aberto leva paralaxe e névoa."""
    t = direcao.lower()
    if re.search(r'close-up|close detail|medium close|extreme close|portrait|face fills', t): return 'rosto'
    if re.search(r'lamp|torch|brazier|flame|fire|candle', t): return 'fogo'
    if re.search(r'medium shot|over-the-shoulder', t): return 'rosto'
    return 'paisagem'


def camera_da_direcao(direcao: str) -> dict:
    """Intenção de câmera escrita na descrição → zoom inicial e final e deriva horizontal (fração da largura)."""
    t = direcao.lower()
    if re.search(r'pull[- ]?back|pulls back|dolly out|zoom out', t): return {'z0': 1.10, 'z1': 1.0, 'pan': 0.0}
    if re.search(r'pan(s|ning)? (to|toward)s? (the )?left|from screen right to (screen )?left', t): return {'z0': 1.06, 'z1': 1.06, 'pan': -0.035}
    if re.search(r'pan(s|ning)? (to|toward)s? (the )?right|from screen left to (screen )?right|tracking', t): return {'z0': 1.06, 'z1': 1.06, 'pan': 0.035}
    if re.search(r'push[- ]?in|dolly in|moves in|creeps in', t): return {'z0': 1.0, 'z1': 1.10, 'pan': 0.0}
    return {'z0': 1.0, 'z1': 1.06, 'pan': 0.0}


def ruido_suave(n: int, semente: int) -> np.ndarray:
    """Série 1D suave (soma de senos de fase fixa): tremor de chama e de câmera reprodutíveis."""
    r = np.random.default_rng(semente)
    t = np.arange(n) / FPS
    return sum(np.sin(2 * np.pi * f * t + r.uniform(0, 2 * np.pi)) * a for f, a in [(0.7, 0.5), (1.9, 0.3), (4.3, 0.2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('imagem')
    ap.add_argument('saida')
    ap.add_argument('--segundos', type=float, default=8)
    ap.add_argument('--direcao', default='')
    ap.add_argument('--tipo', default='auto')
    ap.add_argument('--largura', type=int, default=1920)
    ap.add_argument('--modelo', default=None, help='modelo ONNX de profundidade (opcional)')
    a = ap.parse_args()
    W, H = a.largura, a.largura * 9 // 16
    src = cv2.imread(a.imagem)
    if src is None:
        sys.exit(f'não consegui ler {a.imagem}')
    # Enche o quadro 16:9 por corte central, com folga de 12% para a câmera se mover.
    escala = max(W / src.shape[1], H / src.shape[0]) * 1.12
    img = cv2.resize(src, (int(src.shape[1] * escala), int(src.shape[0] * escala)), interpolation=cv2.INTER_LANCZOS4)
    tipo = tipo_do_plano(a.direcao) if a.tipo == 'auto' else a.tipo
    cam = camera_da_direcao(a.direcao)
    if tipo == 'rosto':
        cam = {'z0': 1.0, 'z1': 1.025, 'pan': 0.0}   # rosto: só "respiração" de câmera
    d = profundidade(img, a.modelo)
    paralaxe = {'paisagem': 0.07, 'fogo': 0.045, 'rosto': 0.0}[tipo]
    print(f'[animatic_pdi] {os.path.basename(a.imagem)} · tipo={tipo} · câmera={cam} · paralaxe={paralaxe}', flush=True)

    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    n = int(a.segundos * FPS)
    treme_x, treme_y = ruido_suave(n, 11) * 0.8, ruido_suave(n, 23) * 0.6        # px de câmera de mão
    chama = ruido_suave(n, 37)
    # Limiarização em HSV: luz quente = tom laranja (H < 28), claro e saturado.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    quente = ((hsv[..., 0] < 28) & (hsv[..., 2] > 150) & (hsv[..., 1] > 70)).astype(np.float32)
    quente = cv2.GaussianBlur(quente, (0, 0), sigmaX=w / 60)[..., None]
    # Névoa: campo de ruído suavizado, deslizando, mais forte no fundo (longe = 1 - d).
    r = np.random.default_rng(5)
    nevoa = cv2.GaussianBlur(r.random((h // 8, (w // 8) * 2)).astype(np.float32), (0, 0), sigmaX=18)
    nevoa = (nevoa - nevoa.min()) / (nevoa.max() - nevoa.min())
    vinheta = (1 - 0.32 * (((xs[:H, :W] - W / 2) / (W / 2)) ** 2 + ((ys[:H, :W] - H / 2) / (H / 2)) ** 2)).clip(0.55, 1)[..., None]

    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', a.saida], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / max(1, n - 1)
        e = t * t * (3 - 2 * t)                                    # ease in-out
        z = cam['z0'] + (cam['z1'] - cam['z0']) * e
        zp = z * (1 + paralaxe * e * d)                            # o perto aproxima mais que o fundo
        dx = cam['pan'] * w * (e - 0.5) * (0.6 + 0.8 * d)          # e desliza mais na deriva
        mx = cx + (xs - cx) / zp - dx + treme_x[i]
        my = cy + (ys - cy) / zp + treme_y[i]
        mx, my = mx.astype(np.float32), my.astype(np.float32)
        q = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        if tipo == 'fogo':
            q *= 1 + quente * (0.10 * chama[i])
        if tipo in ('paisagem', 'fogo'):
            off = int(i * 0.6) % nevoa.shape[1]
            nv = cv2.resize(np.roll(nevoa, -off, axis=1)[:, : w // 8], (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
            q = q * (1 - 0.10 * nv * (1 - d[..., None])) + 210 * 0.10 * nv * (1 - d[..., None])
        oy, ox = (h - H) // 2, (w - W) // 2
        q = q[oy:oy + H, ox:ox + W] * vinheta
        q += np.random.default_rng(1000 + i).normal(0, 5.5, q.shape[:2]).astype(np.float32)[..., None]   # grão
        ff.stdin.write(q.clip(0, 255).astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f'[animatic_pdi] {a.saida} ({n} quadros, {tipo})', flush=True)


if __name__ == '__main__':
    sys.exit(main())
