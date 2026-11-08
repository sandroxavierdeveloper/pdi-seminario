#!/usr/bin/env bash
# Gera os exemplos sintéticos e roda toda a demonstração, gravando em saidas/.
# Uso: ./run_all.sh   (usa o python3 do PATH; ative o venv antes, se houver)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
S=saidas

command -v ffmpeg >/dev/null || { echo "erro: ffmpeg não encontrado no PATH (obrigatório)." >&2; exit 1; }
mkdir -p "$S"

echo "== 1) imagens e vídeo sintéticos"
"$PY" exemplos/gerar_exemplos.py

echo "== 2) detecção de balões (só máscara)"
"$PY" src/limpar_legendas.py exemplos/pagina_sintetica.png --so-mascara --saida "$S"

echo "== 3) retoques locais"
"$PY" src/retoque_local/retoque_pele_roi.py exemplos/pes_sinteticos.png "$S/pes.png" 120 340 200 340 --mascara "$S/pes_mascara.png"
"$PY" src/retoque_local/escurecer_fios_claros.py exemplos/cabelo_sintetico.png "$S/cabelo.png" 200 120 150 90
"$PY" src/retoque_local/gama_e_tom.py exemplos/pagina_sintetica.png "$S/pagina_entardecer.png"
"$PY" src/retoque_local/transferir_galhos_mediana.py exemplos/galhos_v1.png exemplos/galhos_base.png "$S/galhos.png" "$S/galhos_mascara.png"
"$PY" src/retoque_local/transferir_alta_frequencia.py exemplos/textura_sintetica.png "$S/textura.png"

echo "== 4) animatic"
"$PY" src/animatic_pdi.py exemplos/pagina_sintetica.png "$S/animatic.mp4" --segundos 3 --largura 640

echo "== 5) avaliação"
"$PY" src/avaliar_animatic.py exemplos/video_teste.mp4 --saida "$S/avaliacao"
"$PY" src/avaliar_animatic.py "$S/animatic.mp4" --saida "$S/avaliacao_animatic"

echo "Pronto. Resultados em $S/"
