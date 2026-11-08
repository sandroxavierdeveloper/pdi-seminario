"""Funções compartilhadas pelos scripts de retoque local (validação de ROI)."""
import sys


def erro(msg):
    """Mensagem de erro em português e saída com código 2 (uso incorreto)."""
    print(f'erro: {msg}', file=sys.stderr)
    sys.exit(2)


def validar_roi(x0, x1, y0, y1, largura, altura, nome='ROI'):
    """Confere que o retângulo [x0:x1, y0:y1] é não vazio e cabe na imagem."""
    if not (0 <= x0 < x1 <= largura and 0 <= y0 < y1 <= altura):
        erro(f'{nome} (x {x0}:{x1}, y {y0}:{y1}) não cabe na imagem de {largura}x{altura} px. '
             f'Use 0 <= x0 < x1 <= {largura} e 0 <= y0 < y1 <= {altura}.')


def abrir_imagem(caminho):
    """Abre a imagem em RGB ou encerra com mensagem clara."""
    from PIL import Image
    try:
        return Image.open(caminho).convert('RGB')
    except (OSError, ValueError) as e:
        erro(f'não consegui abrir {caminho}: {e}')
