# Modelo do MVP — P3

YOLO11-seg treinado em fotos de drone rotuladas, três classes: `0=construcao`, `1=vegetacao`, `2=via`.

Checkpoint local: `models/mvp_final.pt`; os pesos não são versionados.

**SHA-256:** `0189c55604deaafd81458ce9af39b5850bc5132419f0fb63552ab29a09234c69`.

Dataset P3: 365 fotos; treino 253, validação 70, teste 42.

## Métricas rastreáveis

Fonte: `docs/resultados/numeros_para_o_relatorio.json`, chaves `p3_val.*`, `p3_test.*` e `protocolo.*`. Arquivos originais: `p3_val/resumo.json` e `p3_test/resumo.json`. Valores desta ficha são os da avaliação de 18/11/2026; pequenas diferenças em relação à reprodução anterior ficaram dentro da tolerância de sanidade registrada.

Protocolo: conf=0.25, imgsz=640, retina_masks=True. Pixels acumulados por classe; média das classes com união não nula, sem fundo.

| Split | Classe | IoU | Dice |
|---|---|---:|---:|
| val | construcao | 0.86069428 | 0.92513240 |
| val | vegetacao | 0.77938594 | 0.87601675 |
| val | via | 0.77495043 | 0.87320797 |
| val | Média | 0.80501022 | 0.89145237 |
| test | construcao | 0.84121921 | 0.91376323 |
| test | vegetacao | 0.69668515 | 0.82123092 |
| test | via | 0.71371646 | 0.83294579 |
| test | Média | 0.75054027 | 0.85597998 |

## Limitações

Rótulos pré-anotados por SAM e corrigidos no CVAT; 1768 avisos geométricos em 358 imagens. O teste separa local/sessão, mas não comprova voo independente. Métricas de união de pixels não equivalem a mAP de instâncias nem a medidas de área em metros quadrados. Teste ao vivo do bot e validação de especialista ainda dependem de comprovação.

## Como substituir o modelo

1. Avaliar outro checkpoint com `--pesos` explícito e registrar a decisão usando val; reservar test para confirmação.
2. Parar o bot, copiar o checkpoint aprovado para `models/mvp_final.pt` e conferir o SHA-256, preservando o original.
3. Atualizar esta ficha e reiniciar o bot; o modelo fica em cache no processo.
4. Para outro caminho, definir `MODEL_WEIGHTS` no `.env` local. Caminhos relativos partem da raiz; vazio usa o padrão.

[PREENCHER: link dos pesos]. Condições de publicação e licenças precisam de confirmação pelo responsável.
