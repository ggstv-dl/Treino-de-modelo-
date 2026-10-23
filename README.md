# Treino de modelo: análise de imagens de drone

Modelo de segmentação (YOLO11-seg) que identifica elementos específicos em imagens
capturadas por drones (VANTs). Faz parte do MVP do Trabalho Efetivo Discente de
Análise e Desenvolvimento de Sistemas, UNIBALSAS.

O usuário final vai interagir com o modelo por um bot do Telegram (repositório
[Geo_Map](https://github.com/HugoAlexandreCarvalho/Geo_Map)).

## Estrutura

```
.
├── dataset.yaml        # classes e caminhos do dataset
├── requirements.txt
├── dataset/            # imagens e rótulos (não versionados)
└── src/
    ├── config.py       # constantes e caminhos compartilhados
    ├── train.py        # treino
    ├── evaluate.py     # IoU e Dice por classe
    └── predict.py      # inferência (função analisar(), usada pelo bot)
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Formato YOLO-seg, uma imagem e um `.txt` homônimo:

```
dataset/
  images/train  images/val  images/test
  labels/train  labels/val  labels/test
```

Cada linha do `.txt` é um polígono normalizado: `<id_da_classe> x1 y1 x2 y2 ... xn yn`.
As classes ficam em `dataset.yaml` (a ordem define o id; não altere depois de rotular).
Rotulagem no [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling), exportando em YOLO-seg.

## Uso

Execute sempre a partir da raiz do repositório:

```bash
# 1. Teste de fumaça: 3 épocas em 20% dos dados, só para validar o pipeline
python -m src.train --teste-rapido

# 2. Treino completo (se der erro de memória: --batch 4 ou --modelo yolo11n-seg.pt)
python -m src.train

# 3. Métricas no conjunto de teste (ou --split val)
python -m src.evaluate --json metricas.json

# 4. Inferência em uma imagem
python -m src.predict foto.jpg
```

Os pesos finais ficam em `runs/mvp_drone/weights/best.pt`. Use `--pesos` para apontar outro arquivo.

Uso no código (por exemplo, no bot):

```python
from src.predict import analisar

caminho_imagem, areas = analisar("foto.jpg")   # areas = {"vegetacao": 42.1, ...} em % da imagem
```

## Métricas

- **IoU** = |A∩B| / |A∪B|
- **Dice** = 2|A∩B| / (|A|+|B|)

Calculadas por classe, com os pixels de todo o split acumulados (métrica global),
e reportadas também como média entre classes (mIoU / mDice).

## Status

- [x] Pipeline de treino, avaliação e inferência
- [ ] Dataset rotulado
- [ ] Treino principal
- [ ] Métricas de validação
- [ ] Integração com o bot do Telegram
