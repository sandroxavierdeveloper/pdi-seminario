"""Testes de fumaça do pipeline, sobre as imagens sintéticas de exemplos/.

Rodar na raiz do repositório:  python3 -m pytest
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / 'src'
RETOQUE = SRC / 'retoque_local'
EXEMPLOS = RAIZ / 'exemplos'
sys.path.insert(0, str(SRC))

import avaliar_animatic  # noqa: E402
import limpar_legendas  # noqa: E402


@pytest.fixture(scope='session', autouse=True)
def exemplos():
    """Gera as imagens e o vídeo sintéticos (determinísticos) antes dos testes."""
    subprocess.run([sys.executable, str(EXEMPLOS / 'gerar_exemplos.py')], check=True, capture_output=True)


def rodar(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


def test_baloes_detectados():
    im = cv2.imread(str(EXEMPLOS / 'pagina_sintetica.png'))
    mascara, caixas = limpar_legendas.detectar(im)
    assert len(caixas) == 2
    assert (mascara > 0).any()


def test_avaliar_animatic_acusa_congelamento(tmp_path, capsys):
    sys.argv = ['avaliar_animatic.py', str(EXEMPLOS / 'video_teste.mp4'), '--saida', str(tmp_path)]
    avaliar_animatic.main()
    saida = capsys.readouterr().out
    linha = next(l for l in saida.splitlines() if 'trechos congelados' in l)
    pct = int(linha.split('(')[1].split('%')[0])
    assert 40 <= pct <= 60, linha
    assert (tmp_path / 'medidas.csv').exists() and (tmp_path / 'grafico.png').exists()


@pytest.mark.parametrize('script, args, saidas', [
    ('retoque_pele_roi.py', ['pes_sinteticos.png', '{o}/pes.png', 120, 340, 200, 340, '--mascara', '{o}/m.png'], ['pes.png', 'm.png']),
    ('escurecer_fios_claros.py', ['cabelo_sintetico.png', '{o}/cabelo.png', 200, 120, 150, 90], ['cabelo.png']),
    ('gama_e_tom.py', ['pagina_sintetica.png', '{o}/tom.png'], ['tom.png']),
    ('transferir_galhos_mediana.py', ['galhos_v1.png', 'galhos_base.png', '{o}/g.png', '{o}/gm.png'], ['g.png', 'gm.png']),
    ('transferir_alta_frequencia.py', ['textura_sintetica.png', '{o}/t.png'], ['t.png']),
])
def test_retoques_rodam(script, args, saidas, tmp_path):
    args = [str(EXEMPLOS / a) if str(a).endswith('.png') and '{o}' not in str(a) else str(a).replace('{o}', str(tmp_path)) for a in args]
    r = rodar(RETOQUE / script, *args)
    assert r.returncode == 0, r.stderr
    for s in saidas:
        assert (tmp_path / s).exists()


def test_retoque_pele_altera_pixels(tmp_path):
    r = rodar(RETOQUE / 'retoque_pele_roi.py', EXEMPLOS / 'pes_sinteticos.png', tmp_path / 'o.png', 120, 340, 200, 340)
    assert r.returncode == 0 and int(r.stdout.split()[0]) > 1000


def test_galhos_altera_a_base(tmp_path):
    rodar(RETOQUE / 'transferir_galhos_mediana.py', EXEMPLOS / 'galhos_v1.png', EXEMPLOS / 'galhos_base.png', tmp_path / 'g.png', tmp_path / 'm.png')
    assert (cv2.imread(str(tmp_path / 'm.png'), 0) > 0).sum() > 100


def test_roi_fora_da_imagem_avisa(tmp_path):
    r = rodar(RETOQUE / 'retoque_pele_roi.py', EXEMPLOS / 'pes_sinteticos.png', tmp_path / 'o.png', 120, 5000, 200, 340)
    assert r.returncode != 0
    assert 'não cabe na imagem' in r.stderr
