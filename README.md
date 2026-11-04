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
    ├── predict.py      # inferência (função analisar(), usada pelo bot)
    ├── validar_dataset.py   # conferência de imagens e polígonos
    ├── selecionar_imagens.py # seleção das fotos próprias
    └── bot.py          # bot assíncrono do Telegram
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
### Gerando o dataset a partir do Semantic Drone Dataset

Baixe o [Semantic Drone Dataset](https://www.kaggle.com/datasets/awsaf49/semantic-drone-dataset)
(imagens, máscaras e `class_dict.csv`). Primeiro confira se o script entende o download
(não grava nada):

```bash
python -m src.preparar_dataset --imagens CAMINHO/images --mascaras CAMINHO/masks \
    --class-dict CAMINHO/class_dict.csv --inspecionar
```

Depois gere o dataset (redimensiona, agrupa as classes, converte em polígonos e divide 70/20/10 por imagem):

```bash
python -m src.preparar_dataset --imagens CAMINHO/images --mascaras CAMINHO/masks \
    --class-dict CAMINHO/class_dict.csv
```

O agrupamento de classes (por exemplo, `roof` e `wall` viram `construcao`) fica na constante
`MAPEAMENTO` do script e precisa usar os mesmos nomes do `dataset.yaml`.
`--sobrescrever` refaz somente `images/{train,val,test}` e `labels/{train,val,test}`;
splits extras, como `test_proprio`, são preservados. Confira os dados antes de refazer.

### Imagens próprias

Rotulagem no [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling), exportando em YOLO-seg.

Prepare uma seleção a partir de uma pasta local (a busca inclui suas subpastas):

```bash
python -m src.selecionar_imagens --origem PASTA_DAS_FOTOS --saida ~/dados/drone_proprio \
    --n-teste 40 --n-reserva 60 --lado-maior 1024 --limiar-duplicata 6 --seed 42
```

O script descarta os 10% menos nítidos e quase-duplicatas por dHash, agrupa capturas
em voos por intervalos maiores que dez minutos e seleciona imagens diversas por cor
e dHash. Prefere voos diferentes para teste e reserva; o relatório informa qualquer
compartilhamento e insuficiência de imagens. Sem data EXIF, usa mtime, que deve ser
conferido antes de interpretar os grupos como voos reais. A saída deve estar vazia.

Os JPEGs ficam em `teste_para_rotular/` e `reserva_treino/`, com nomes sequenciais;
`mapeamento.csv` relaciona originais, novos nomes, datas e voos. O relatório fica em
`relatorio_selecao.md`. O OpenCV grava JPEG qualidade 95 com lado maior de até 1024 px
e remove os metadados EXIF, inclusive GPS. Os originais são preservados.
Essas fotos e seus rótulos ficam fora do Git. A reserva não deve ser incorporada ao
teste após ajustes de modelo.

Depois de rotular o teste, copie os pares para `dataset/images/test_proprio/` e
`dataset/labels/test_proprio/`, sem misturá-los ao treino, e valide:

```bash
python -m src.validar_dataset                     # todos os splits em dataset/images
python -m src.validar_dataset --split test_proprio
python -m src.validar_dataset --split val test
```

O validador verifica pares, classes, polígonos normalizados e imagens repetidas
entre splits por nome ou conteúdo. Imprime contagens por classe e retorna código
diferente de zero se houver erro; rótulos vazios são contados e permitidos.

## Limpeza das imagens do drone

Para limpar todas as fotos antes de selecionar ou rotular, use uma pasta de saída
separada. A busca inclui subpastas e os originais são somente lidos. Comece pela simulação:

```bash
python -m src.limpar_imagens --origem PASTA_DAS_FOTOS --saida ~/dados/drone_limpo_1024 --simular
```

Confira `relatorio_limpeza.md`, `descartadas.csv`, `montagem_descartadas.jpg` e
`montagem_mantidas.jpg`. Se o descarte total superar 30% ou um motivo superar 15%,
interrompa e revise os resultados antes de autorizar a gravação ou alterar limiares.
Depois de aprovar a simulação, execute sem `--simular`:

```bash
python -m src.limpar_imagens --origem PASTA_DAS_FOTOS --saida ~/dados/drone_limpo_1024 \
    --lado-maior 1024 --qualidade 95 --limiar-borrao 0.35 \
    --limiar-duplicata 5 --lado-minimo 640 --workers 4 --seed 42
```

As etapas são aplicadas nesta ordem:

1. Leitura com Pillow, correção da orientação EXIF e conversão para RGB; arquivos
   corrompidos recebem o motivo `ilegivel`.
2. Descarte de imagens com qualquer lado menor que 640 px (`muito_pequena`).
3. Média de cinza menor que 25 ou maior que 235 (`exposicao`), seguida de desvio
   padrão menor que 10 (`sem_conteudo`). Assim, uma imagem branca é exposição.
4. Variância do Laplaciano calculada após reduzir o lado maior para até 1024 px,
   sem ampliar e independentemente de `--lado-maior`. O limite de desfoque é
   `--limiar-borrao` vezes a mediana das imagens que passaram nas etapas anteriores;
   valores abaixo do limite recebem `borrada`.
5. Comparação de dHash de 64 bits com todas as fotos mantidas: distância de Hamming
   até 5 indica `duplicata`. As mais nítidas têm prioridade; o CSV informa a cópia mantida.
6. Redução com INTER_AREA para lado maior de até 1024 px, sem ampliar, e JPEG qualidade 95.
   A gravação **remove EXIF, inclusive GPS**, dos arquivos finais.
7. Ordenação por DateTimeOriginal do EXIF ou mtime, com desempate pelo nome original,
   e nomes contínuos `drone_0001.jpg`, `drone_0002.jpg` etc.

`--workers` controla os processos de leitura e gravação; os pixels são liberados a
cada imagem e apenas metadados permanecem na memória principal. O progresso aparece
a cada 25 fotos. `--seed` torna determinísticas as amostras das montagens.

A simulação grava somente CSVs, relatório e montagens, sem os JPEGs individuais.
O mapeamento contém nomes e dimensões planejados; os bytes finais ficam vazios.
Os relatórios da simulação podem ser reutilizados na execução real na mesma pasta.
JPEGs individuais e arquivos desconhecidos já existentes bloqueiam a execução;
nenhuma imagem é sobrescrita. Registre a contagem e a soma dos bytes dos originais
em `_antes.txt` e confira novamente ao final. Mantenha fotos e relatórios fora do Git.

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

O experimento antigo `mvp_drone` foi preservado. O treino principal desta etapa usa
um nome explícito; informe também os pesos explicitamente na avaliação e inferência:

```bash
python -m src.train --nome mvp_drone_principal --epochs 50 --patience 20 --batch 8
python -m src.train --nome mvp_drone_principal --retomar
python -m src.evaluate --split test_proprio --pesos runs/mvp_drone_principal/weights/best.pt
python -m src.predict foto.jpg --pesos runs/mvp_drone_principal/weights/best.pt
```

`--retomar` exige `runs/<nome>/weights/last.pt` e continua com `resume=True`, usando
os parâmetros e o total de épocas salvos no checkpoint. Não combina com
`--teste-rapido` e não serve para acrescentar épocas a um treino já concluído.
No Colab, preserve toda a pasta do experimento no Drive e restaure-a em `runs/<nome>/`
antes de retomar, junto com o dataset nos caminhos do treino original.

`evaluate --split` aceita nomes livres: usa o caminho de imagens do `dataset.yaml`
quando definido; caso contrário, procura `dataset/images/<split>`. Splits extras
usam rótulos em `dataset/labels/<split>`. Se a pasta não existir, o erro lista os
splits disponíveis. Os caminhos e o comportamento de `train`, `val` e `test` permanecem iguais.

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

Na avaliação de 10 fotos próprias em 29/10/2026, o modelo obteve **mIoU de
51,77% e mDice de 67,59%**. Os parâmetros, resultados por classe e limitações
estão no [relatório da avaliação](docs/avaliacao_fotos_proprias.md).

## Status

- [x] Pipeline de treino, avaliação e inferência
- [x] Script de preparação do dataset
- [x] Dataset gerado e conferido (280 treino, 80 validação, 40 teste)
- [x] Treino parcial de 25 épocas e métricas de validação/teste registradas
- [ ] Treino principal concluído (50 épocas solicitadas; experimento `mvp_drone_principal` iniciado)
- [x] Validação estrutural, splits adicionais, retomada e seleção de fotos próprias
- [x] Bot do Telegram implementado
- [ ] Teste ao vivo do bot e correção da pendência de arquivo inválido (`pi-heif`)
- [ ] Rotulagem e avaliação do conjunto `test_proprio`

## Bot do Telegram

Crie um bot conversando com o **@BotFather** no Telegram e guarde o token em um
arquivo `.env` local, usando `.env.example` como modelo. Preencha
`TELEGRAM_BOT_TOKEN` e, para usar o treino parcial, defina
`MODEL_WEIGHTS=runs/mvp_drone_parcial/weights/best.pt`. Se `MODEL_WEIGHTS` ficar
vazio, serão usados os pesos padrão de `src.config`. O `.env` está no `.gitignore`;
nunca inclua seu token em mensagens, logs ou commits.
Para o treino principal, após existir o checkpoint, use
`MODEL_WEIGHTS=runs/mvp_drone_principal/weights/best.pt`.

Com a `.venv` ativa, instale as dependências e inicie:

```bash
pip install -r requirements.txt
python -m src.bot
```

Envie `/start` ou `/ajuda` ao bot e depois uma foto de drone. Para preservar a
resolução, envie a imagem como documento JPG ou PNG. O bot devolve a imagem
anotada e os percentuais de área por classe. Esses percentuais são relativos à
imagem, não medidas em metros quadrados. O processo precisa permanecer ativo
para responder às mensagens.
