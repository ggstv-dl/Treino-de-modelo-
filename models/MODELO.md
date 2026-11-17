# Modelo do MVP — P3

YOLO11-seg treinado em fotos de drone rotuladas, 3 classes.

Checkpoint aprovado: `mvp_drone_proprio_p3/weights/best.pt`, copiado para `models/mvp_final.pt`.

**SHA-256:** `0189c55604deaafd81458ce9af39b5850bc5132419f0fb63552ab29a09234c69`.

Classes, na ordem: `0=construcao`, `1=vegetacao`, `2=via`.
Dataset P3: **365 fotos**; treino **253**, validação **70**, teste **42**.

## Métricas reproduzidas em 16/11/2026

IoU e Dice calculados sobre pixels acumulados por classe; médias entre as classes com união não nula, sem fundo. `conf=0.25`, `imgsz=640`, `retina_masks=True`.

| Split | Classe | IoU | Dice |
|---|---|---:|---:|
| val | construcao | 0.86073071 | 0.92515344 |
| val | vegetacao | 0.77936893 | 0.87600600 |
| val | via | 0.77493819 | 0.87320020 |
| val | **Média** | **0.80501261** | **0.89145321** |
| test | construcao | 0.84125407 | 0.91378380 |
| test | vegetacao | 0.69672331 | 0.82125743 |
| test | via | 0.71371837 | 0.83294709 |
| test | **Média** | **0.75056525** | **0.85599611** |

Fontes: `dia16/baseline/metricas_p3_reproducao_val.json` e `dia16/baseline/metricas_p3_reproducao_test.json` das evidências do projeto.

## Limitações

As avaliações usam uma divisão local do conjunto de fotos próprias. Não comprovam generalização a novos locais, voos ou ângulos. Vegetação é a classe com menor IoU no teste (0,69672331). A triagem das anotações não substitui a revisão humana completa. Percentuais representam pixels da imagem, sem conversão para metros quadrados. O teste ao vivo do bot ainda depende de execução humana.

## Como substituir o modelo

1. Avaliar o novo checkpoint com `--pesos` explícito e registrar a decisão e seu SHA-256.
2. Parar o bot, copiar o checkpoint aprovado para `models/mvp_final.pt` e conferir o SHA-256; não mover o original.
3. Atualizar este documento e reiniciar o bot, pois o modelo fica em cache por processo.
4. Para experimentar outro checkpoint local, definir `MODEL_WEIGHTS` no `.env`; caminhos relativos partem da raiz do projeto. Com o valor vazio, vale `models/mvp_final.pt`.

Os pesos são artefatos locais ignorados pelo Git. Este documento é versionado; o arquivo `.pt` não é incluído nos commits.
