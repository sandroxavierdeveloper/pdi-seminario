# Slides da apresentação — HTML

Versão atual dos slides do seminário em HTML autocontido:
**`slides_atualizados.html`** (abrir no navegador; as mídias estão em `img/`).

## Estrutura (6 slides)

1. **Capa** — tese econômica: vídeo de IA custa ~R$ 0,50/s por tentativa,
   ou ~R$ 100 mil de setup local (GPU ~R$ 40 mil + fine-tuning contínuo de
   LLM para agentes) rodando semanas antes do primeiro render.
2. **Motivação** — as duas saídas caras (API × setup local); a resposta do
   pipeline (animatic barato + juiz automático + só o aprovado vai ao modelo
   de vídeo); exemplo real de correção pontual por código (antes → máscara →
   depois) com a nota "PDI ou Photoshop? PDI dos dois jeitos — a diferença é
   quem aplica".
3. **O dataset que justifica a automação** — ~21 mil quadros de cenas,
   300 imagens-base (domínio público), ~1,9 mil referências visuais
   (filmes 1903–2013), ~6,5 mil vídeos; +1.700 h de retoque manual evitadas.
4. **Pipeline** — 7 etapas: imagem sem direitos autorais → IA gera →
   PDI corrige → PDI anima → PDI julga → **motor gráfico renderiza os
   quadros (batchmode)** → **IA finaliza o vídeo** (ordem correta:
   render ANTES do modelo de vídeo). Inclui faixa entrada → animatic → MP4.
5. **Ferramentas do PDI** — 6 ferramentas em linguagem simples com capítulos
   do livro-texto + gráfico real de avaliação do `avaliar_animatic.py`
   em faixa total.
6. **Conclusão** — takeaways e linha-síntese do híbrido.

## Decisões editoriais

- **Fonte base 18px** (sala ampla); mínimo absoluto ~14px só em itens decorativos.
- **Sem marcas comerciais**: o motor de render é citado apenas como "motor
  gráfico"; a entrada é "imagem sem direitos autorais" (não se nomeia o
  artista da gravura); a LLM de fine-tuning é citada genericamente.
- **Sem fórmulas**: os limiares e transformações ficam no código-fonte
  (`src/`); os slides trabalham com a ideia em linguagem leiga.
- Volumes do dataset em ordem de grandeza (~) — medidos em disco, crescem a
  cada rodada do pipeline.
