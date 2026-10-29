# Avaliação nas fotos próprias — 29/10/2026

O modelo já treinado foi avaliado em 10 fotos próprias de drone, com 176
polígonos de referência: 92 de construção, 72 de vegetação e 12 de via.
O resultado foi **mIoU de 51,77% e mDice de 67,59%**. As comparações visuais
mostram erros de classificação de telhados como vias e telhados sem detecção
de construção.

## Configuração

- Pesos: `runs/mvp_drone/weights/best.pt` (YOLO11s-seg).
- SHA-256 dos pesos: `5273d0c771806307aabace260495866f5d088859851abf0a4a7490b3fde16ff1`.
- Imagens: `dataset/rotulagem_propria/images/`.
- Rótulos exportados: `dataset/labels/images/`.
- Classes: `0 construcao`, `1 vegetacao`, `2 via`.
- Resolução de entrada: `imgsz=640`; limiar de confiança: `conf=0.25`.
- Execução na CPU, com uma imagem por vez e duas threads do PyTorch.

A avaliação reutilizou `carregar_ground_truth`, `prever_mascaras` e
`calcular_metricas` de [src/evaluate.py](../src/evaluate.py), por meio de um
script temporário que apontava para as pastas acima. A CLI atual de
`src.evaluate` usa os splits de `dataset.yaml` e não seleciona esta amostra
diretamente.

As máscaras previstas foram avaliadas na resolução original, preservando
buracos e resolvendo sobreposições pela confiança da instância. Interseções
e áreas foram acumuladas nas 10 imagens antes do cálculo de IoU e Dice por
classe; a média é aritmética entre as três classes.

## Resultados

| Classe | IoU | Dice |
| --- | ---: | ---: |
| Construção | 52,56% | 68,90% |
| Vegetação | 64,15% | 78,16% |
| Via | 38,60% | 55,70% |
| **Média** | **51,77%** | **67,59%** |

Os [resultados completos em JSON](metricas_fotos_proprias.json) incluem as
métricas por imagem, contagens de pixels e hashes dos arquivos avaliados.
Os hashes permitem identificar alterações posteriores nas imagens, nos
rótulos ou nos pesos.

Os 10 pares imagem/TXT foram conferidos antes da inferência: IDs inteiros de
0 a 2, ao menos três pontos por polígono, número par de coordenadas e valores
finitos entre 0 e 1. As contagens de polígonos e classes exportadas coincidiram
com os JSON do X-AnyLabeling. Os arquivos avaliados e os pesos mantiveram os
mesmos hashes após a execução. Nenhum treinamento foi realizado.

## Limitações e próximo passo

A amostra foi selecionada para diagnóstico e contém fotos do mesmo voo;
os resultados não representam todos os cenários de uso. A referência foi
rotulada com assistência do SAM e depende da revisão humana de todas as
regiões visíveis. Pequenas sobreposições entre classes na referência foram
preservadas: 7.120 pixels, equivalentes a 0,0078% dos pixels das 10 imagens.

O próximo passo recomendado é coletar e rotular novas cenas ou voos para
ajustar o modelo, mantendo estas 10 fotos reservadas para avaliação e
evitando cenas sobrepostas entre treino e teste.

As imagens, rótulos, pesos e visualizações permanecem fora do versionamento.
