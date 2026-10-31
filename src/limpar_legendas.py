#!/usr/bin/env python3
"""
limpar_legendas.py: detecta balões de fala e caixas de legenda em páginas de
quadrinhos e, opcionalmente, reconstrói o que havia atrás delas.

Técnicas de PDI usadas (unidades da disciplina entre parênteses):
  1. Limiarização em HSV: baixa saturação e alto brilho isolam o "miolo" claro
     do balão; baixo brilho isola a tinta (contorno e letras).        (6, 7)
  2. Morfologia: fechamento 3x3, erosão e dilatação com elemento 5x5.  (5)
  3. Componentes conexos e preenchimento de contornos: o contorno externo
     de cada componente é preenchido (as letras são "buracos").        (5, 6)
  4. Descritores de região: área, largura, altura, fração de tinta e
     alinhamento das letras em linhas decidem o que é balão.           (8)
  5. União de máscaras e mistura ponderada, com máscara suavizada
     por filtro gaussiano.                                            (2)
  6. Opcional: preenchimento por inpainting neural (LaMa). É uma rede
     neural usada como caixa-preta; a máscara que a alimenta é PDI clássico.

Assinatura que só balão e caixa de legenda têm (e céu, roupa branca e calhas
entre painéis não têm): MIOLO CLARO, FECHADO POR CONTORNO ESCURO, COM LETRAS
DENTRO.

Uso:
  python3 limpar_legendas.py <pasta-ou-imagem>... [--so-mascara] [--saida PASTA]
  --so-mascara   só detecta; grava as máscaras em <saida>/mascaras
  --saida PASTA  onde gravar os resultados (padrão: ./saidas). Os arquivos
                 originais NUNCA são alterados.
"""
import os
import sys
import time

import cv2
import numpy as np

# ── Parâmetros da detecção ────────────────────────────────────────────────
SAT_MAX = 135             # saturação abaixo disto = claro "de papel"
VAL_MIN = 190             # brilho acima disto = claro
ESCURO_MAX = 128          # brilho abaixo disto = tinta (contorno, letra)
AREA_MIN = 1500           # px² mínimos do componente claro
LARGURA_MIN = 40
ALTURA_MIN = 20
PREENCHIMENTO_MIN = 0.25  # área preenchida / retângulo: corta calhas e céu recortado
TEXTO_MIN, TEXTO_MAX = 0.04, 0.50   # fração de tinta dentro do miolo
LETRAS_MIN = 6
ANEL_MIN = 0.38           # fração de tinta na coroa de 4 px em volta do contorno
DILATACAO = 7             # px além do contorno: engole a borda e o ringing do JPEG
LARGURA_DO_LAMA = 1024    # largura de trabalho do inpainting neural


def parece_texto(letras):
    """Letras vêm em LINHAS: componentes de altura parecida e alinhados em y.
    Um rosto claro com cabelo escuro passa pelos outros critérios (olhos,
    sobrancelhas e boca viram "letras"), mas não forma linha. Exige 2 linhas
    com pelo menos 3 letras, ou 1 linha com pelo menos 5."""
    alturas = np.array([l[3] for l in letras], float)
    hm = float(np.median(alturas))
    if hm <= 0:
        return False
    regulares = [l for l in letras if 0.5 * hm <= l[3] <= 1.6 * hm]
    if len(regulares) < LETRAS_MIN:
        return False
    centros = sorted(l[1] + l[3] / 2.0 for l in regulares)
    linhas, atual = [], [centros[0]]
    for c in centros[1:]:
        if c - atual[-1] <= 0.6 * hm:
            atual.append(c)
        else:
            linhas.append(atual)
            atual = [c]
    linhas.append(atual)
    cheias = [len(l) for l in linhas if len(l) >= 3]
    return len(cheias) >= 2 or (len(cheias) >= 1 and max(cheias) >= 5)


def detectar(im):
    """Devolve (máscara uint8 0/255, lista de caixas [x, y, w, h])."""
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Limiarização: comparações em S e V geram máscaras binárias.
    claro = ((hsv[..., 1] < SAT_MAX) & (hsv[..., 2] > VAL_MIN)).astype(np.uint8) * 255
    escuro = (hsv[..., 2] < ESCURO_MAX).astype(np.uint8) * 255
    # Fechamento 3x3: só cola o anti-aliasing das letras, sem atravessar o contorno.
    claro = cv2.morphologyEx(claro, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Antes de erodir, as LETRAS (componentes escuros pequenos) entram no claro;
    # sem isso a erosão retalharia o miolo entre as letras. A erosão de 2 px
    # fecha frestas do contorno por onde o miolo "vazaria" para o céu claro.
    nl, rotl, stl, _ = cv2.connectedComponentsWithStats(escuro, connectivity=8)
    pequenos = np.zeros(nl, np.uint8)
    for k in range(1, nl):
        if stl[k][4] <= 900 and stl[k][3] <= 60 and stl[k][2] <= 60:
            pequenos[k] = 1
    letras_px = pequenos[rotl] * 255
    # Limitação conhecida: balão sobre nuvem branca com contorno cinza-claro
    # faz o miolo vazar para a nuvem; a página vira um componente só e não é detectada.
    claro = cv2.erode(claro | letras_px, np.ones((5, 5), np.uint8))
    n, rot, st, _ = cv2.connectedComponentsWithStats(claro, connectivity=4)
    mascara = np.zeros((H, W), np.uint8)
    caixas = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < AREA_MIN or w < LARGURA_MIN or h < ALTURA_MIN or w > W * 0.7 or h > H * 0.5:
            continue
        comp = (rot[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cheio = np.zeros_like(comp)
        cv2.drawContours(cheio, cnts, -1, 1, thickness=-1)   # preenche: letras são buracos
        area_cheia = int(cheio.sum())
        if area_cheia / float(w * h) < PREENCHIMENTO_MIN:
            continue
        miolo = escuro[y:y + h, x:x + w] & (cheio * 255)
        fracao_texto = miolo.sum() / 255.0 / max(1, area_cheia)
        if not (TEXTO_MIN <= fracao_texto <= TEXTO_MAX):
            continue
        nl, _, stl, _ = cv2.connectedComponentsWithStats(miolo, connectivity=8)
        letras = [stl[k] for k in range(1, nl) if 12 <= stl[k][4] <= 900 and stl[k][3] <= 0.6 * h]
        if len(letras) < LETRAS_MIN or not parece_texto(letras):
            continue
        pad = 8
        y0, y1, x0, x1 = max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)
        cheio_g = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cheio_g[y - y0:y - y0 + h, x - x0:x - x0 + w] = cheio
        # Devolve os 2 px da erosão antes de medir o anel de contorno.
        cheio_g = cv2.dilate(cheio_g, np.ones((5, 5), np.uint8))
        anel = cv2.dilate(cheio_g, np.ones((9, 9), np.uint8)) - cheio_g
        fracao_anel = (escuro[y0:y1, x0:x1] & (anel * 255)).sum() / 255.0 / max(1, anel.sum())
        if fracao_anel < ANEL_MIN:
            continue
        cheio_d = cv2.dilate(cheio_g, np.ones((DILATACAO * 2 + 1, DILATACAO * 2 + 1), np.uint8))
        mascara[y0:y1, x0:x1] |= cheio_d * 255            # união (OR) das máscaras
        caixas.append([int(x), int(y), int(w), int(h)])
    return mascara, caixas


_lama = None


def preencher(im, mascara):
    """Inpainting neural (LaMa) só dentro da máscara, com borda suave."""
    global _lama
    if _lama is None:
        import torch
        from simple_lama_inpainting import SimpleLama
        # O modelo vem gravado com tensores CUDA; em torch só-CPU o carregamento
        # precisa de map_location='cpu', e o pacote não repassa esse argumento.
        original = torch.jit.load
        torch.jit.load = lambda caminho, *a, **k: original(caminho, map_location='cpu')
        try:
            _lama = SimpleLama(device=torch.device('cpu'))
        finally:
            torch.jit.load = original
    from PIL import Image
    # O modelo rende melhor perto de 1024 px de largura; o resultado volta ao
    # tamanho original (reamostragem com interpolação) e entra só na máscara.
    H, W = im.shape[:2]
    esc = min(1.0, LARGURA_DO_LAMA / float(W))
    pe = cv2.resize(im, (int(W * esc), int(H * esc)), interpolation=cv2.INTER_AREA) if esc < 1 else im
    me = cv2.resize(mascara, (pe.shape[1], pe.shape[0]), interpolation=cv2.INTER_NEAREST) if esc < 1 else mascara
    rgb = Image.fromarray(cv2.cvtColor(pe, cv2.COLOR_BGR2RGB))
    out = np.array(_lama(rgb, Image.fromarray(me)).convert('RGB'))[:pe.shape[0], :pe.shape[1]]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if esc < 1:
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    # Mistura ponderada: a máscara suavizada por filtro gaussiano vira o peso alfa.
    alpha = cv2.GaussianBlur(mascara.astype(np.float32) / 255.0, (0, 0), 3)[..., None]
    return (out * alpha + im * (1 - alpha)).astype(np.uint8)


def paginas(args):
    for a in args:
        if os.path.isdir(a):
            for f in sorted(os.listdir(a)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    yield os.path.join(a, f)
        else:
            yield a


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    so_mascara = '--so-mascara' in sys.argv
    saida = './saidas'
    if '--saida' in sys.argv:
        saida = sys.argv[sys.argv.index('--saida') + 1]
        args = [a for a in args if a != saida]
    if not args:
        print(__doc__)
        sys.exit(2)
    os.makedirs(os.path.join(saida, 'mascaras'), exist_ok=True)
    for p in paginas(args):
        t0 = time.time()
        im = cv2.imread(p)
        nome = os.path.basename(p)
        if im is None:
            print(f'{nome} · não consegui ler', flush=True)
            continue
        mascara, caixas = detectar(im)
        pct = 100.0 * (mascara > 0).sum() / mascara.size
        base = os.path.splitext(nome)[0]
        if caixas:
            cv2.imwrite(os.path.join(saida, 'mascaras', base + '.png'), mascara)
        if caixas and not so_mascara:
            cv2.imwrite(os.path.join(saida, base + '_limpa.png'), preencher(im, mascara))
        dt = round(time.time() - t0, 1)
        sufixo = ' · só máscara' if so_mascara else (' · limpa' if caixas else '')
        print(f'{nome} · {len(caixas)} caixa(s) · {pct:.1f}% da área · {dt}s{sufixo}', flush=True)


if __name__ == '__main__':
    main()
