# Segmentação de imagens de drone — TED UNIBALSAS

## Visão geral

Atividades de mapeamento e topografia podem exigir identificação manual de elementos em imagens aéreas, um processo custoso e sujeito a erro. Este projeto implementa um MVP de segmentação de construções, vegetação e vias com YOLO11-seg, integrado a um bot do Telegram que devolve imagem anotada e percentuais de pixels por classe.

O trabalho integra o TED de Análise e Desenvolvimento de Sistemas da UNIBALSAS. O pipeline, o bot e os testes estão implementados; operação ao vivo, consentimento para divulgação das fotos e validação por especialista ainda precisam de comprovação.

## Resultados

Valores da avaliação de 18/11/2026 em [docs/resultados/numeros_para_o_relatorio.json](docs/resultados/numeros_para_o_relatorio.json). Os intervalos usam bootstrap pareado de imagens, não voos independentes.

### P3 — fotos próprias

| Split | Imagens | mIoU [IC95%] | mDice [IC95%] |
|---|---:|---|---|
| val | 70 | 80,5010% [78,7895%–82,0177%] | 89,1452% [88,0460%–90,0889%] |
| test | 42 | 75,0540% [70,7962%–78,9450%] | 85,5980% [82,5191%–88,1660%] |

Modelo P3, SHA8 `0189c556`; chaves `p3_val.*`, `p3_test.*` e `comparacao_*.metricas.*.B`.

### Baseline — principal público nas mesmas fotos próprias

| Split | Imagens | mIoU [IC95%] | mDice [IC95%] |
|---|---:|---|---|
| val | 70 | 45,4536% [42,9338%–48,0171%] | 60,8805% [58,3306%–63,3901%] |
| test | 42 | 47,6342% [43,4130%–51,7222%] | 64,3075% [60,2844%–67,8816%] |

Principal público, SHA8 `0692f705`; chaves `principal_proprio_*` e `comparacao_*.metricas.*.A`. Seu treino público foi concluído; não é o treino P3.

| Split | P3 − baseline em mIoU (p.p.) [IC95%] | P3 − baseline em mDice (p.p.) [IC95%] |
|---|---|---|
| val | 35,0475 [32,8428–37,3505] | 28,2648 [26,0116–30,7234] |
| test | 27,4198 [23,3001–32,0620] | 21,2905 [17,9868–25,1531] |

**Comparações só valem dentro do mesmo conjunto.** IoU/Dice medem concordância com os rótulos disponíveis, pré-anotados por SAM e corrigidos no CVAT, não verdade de campo validada por especialista. O test é separado por local/sessão, mas não é um voo independente. Não foi escolhido limiar, checkpoint ou configuração pelo test. Consulte [tabelas, fontes e limitações](docs/resultados/).

## Demonstração

- [Guia para executar a demonstração](docs/DEMONSTRACAO.md).
- [PREENCHER: link do bot ou da demo].
- [PREENCHER: link do vídeo].

## Estrutura do repositório

```text
.
├── dataset.yaml             # classes e caminhos do dataset local
├── requirements.txt
├── .env.example             # campos vazios, sem credenciais
├── src/                     # preparo, treino, avaliação, inferência e bot
├── tests/                   # unittest com imagens sintéticas e stubs
├── models/MODELO.md         # identificação do P3; pesos fora do Git
├── docs/                    # fluxo, ambiente, créditos e diagnósticos históricos
│   ├── resultados/          # fonte numérica e tabelas
│   └── figuras/             # gráficos sem fotos
├── scripts/                 # roteiros históricos, identificadores anonimizados
├── configs/                 # configuração do experimento histórico
└── dataset/.gitkeep          # só marcador; imagens e rótulos não versionados
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows, ativar com `.venv\Scripts\activate`. A build de torch deve ser adequada à CPU/GPU da máquina; não há versão de torch ou CUDA fixada pelo projeto. Para repetir o ambiente de teste, consulte [AMBIENTE_TESTADO.md](docs/AMBIENTE_TESTADO.md). Instalar dependências não fornece dados privados nem checkpoints.

## Uso

Execute na raiz, com a `.venv` ativa. Os caminhos abaixo são exemplos locais; prepare suas próprias pastas autorizadas. Os comandos de treino não foram executados nesta revisão.

### Inferência

```bash
python -m src.predict foto.jpg --pesos models/mvp_final.pt --saida resultado.jpg --conf 0.25 --imgsz 640
```

`predict.analisar` retorna caminho anotado e percentuais da imagem; não calcula metros quadrados. `--pesos`, `--conf` e `--imgsz` permitem indicar o protocolo explicitamente.

### Avaliação

```bash
python -m src.evaluate --pesos models/mvp_final.pt --split val --json RESULTADOS/metricas_val.json --conf 0.25 --imgsz 640
```

Passe sempre `--pesos`. O avaliador usa as pastas do `dataset.yaml`; `--split` também aceita nomes extras, como `test_proprio`. Para o pacote privado P3 externo ao dataset padrão, use as pastas explícitas de `analise_erros avaliar`. Não copie fotos de treino para testar o modelo.

### Análise por pixel e comparação pareada

```bash
python -m src.analise_erros avaliar --pesos models/mvp_final.pt --imagens DADOS_P3/images/val --rotulos DADOS_P3/labels/val --saida RESULTADOS/p3_val --imgsz 640 --conf 0.25 --grupos DADOS_P3/sanidade_divisao.json
python -m src.analise_erros comparar --a RESULTADOS/baseline_val --b RESULTADOS/p3_val --saida RESULTADOS/comparacao_val --n-boot 2000 --seed 42
```

Avalie previamente o baseline nas mesmas imagens/rótulos para obter `RESULTADOS/baseline_val`. A comparação aborta para listas diferentes. O agrupamento usa o JSON e, quando necessário, `selecao.csv` adjacente. O mapa de locais reais fica somente na pasta local de evidências, em `resultados_finais/scripts/mapa_locais_INTERNO.csv`, e não deve ser publicado. A matriz exclui pixels com sobreposição entre classes e registra a fração ambígua; IoU/Dice mantêm o protocolo de união de áreas.

### Treino

```bash
python -m src.train --nome experimento_local --epochs 50 --patience 20 --batch 4 --imgsz 640
python -m src.train --nome experimento_local --retomar
```

`src.train` usa o dataset do projeto; não é o lançador externo original do P3. `--modelo` escolhe os pesos pré-treinados. `--retomar` exige `runs/<nome>/weights/last.pt` e preserva o plano salvo; não acrescenta épocas a um treino concluído. Use um nome novo para novos experimentos.

### Bot do Telegram

Crie um bot no @BotFather e copie `.env.example` para `.env` no editor. Preencha `TELEGRAM_BOT_TOKEN` somente nesse arquivo local. `MODEL_WEIGHTS` é opcional: vazio usa `models/mvp_final.pt`; caminhos relativos partem da raiz. Nunca cole tokens em chat, comandos, prints ou commits.

```bash
python -m src.bot --verificar
python -m src.bot
```

`--verificar` faz uma consulta ao nome do bot e encerra. O segundo comando mantém o processo ativo até Ctrl+C; sua execução e as conversas ao vivo dependem do responsável. `/start` e `/ajuda` orientam a enviar uma foto aérea ou JPG/PNG como arquivo. Os resultados são estimativas; outros ângulos podem produzir erros. O bot rejeita arquivos inválidos, acima de seu limite e imagens muito pequenas. HEIC não é suportado.

## Modelo

Consulte [models/MODELO.md](models/MODELO.md) para SHA-256, métricas, limitações e substituição. **Pesos não estão no Git.**

[PREENCHER: link dos pesos]. A publicação depende da decisão do responsável sobre licenças e condições de uso.

## Dados

Para usar o Semantic Drone Dataset, obter imagens, máscaras e dicionário de classes na fonte oficial/Kaggle e conferir as restrições de licença. Não redistribuir o dataset nem presumir autorização comercial. As fotos próprias são privadas e não são distribuídas.

```bash
python -m src.preparar_dataset --imagens DADOS_PUBLICOS/images --mascaras DADOS_PUBLICOS/masks --class-dict DADOS_PUBLICOS/class_dict.csv --inspecionar
python -m src.validar_dataset --split train val test
```

Após revisar a inspeção, execute o preparo sem `--inspecionar` para converter máscaras em polígonos. `--sobrescrever` refaz apenas train/val/test e preserva splits extras; a divisão por imagem do dataset público não é o agrupamento por local do P3. `--lado-maior`, `--area-min` e `--epsilon` controlam redução e conversão.

Para dados próprios autorizados, em saídas separadas dos originais:

```bash
python -m src.limpar_imagens --origem FOTOS_ORIGINAIS --saida DADOS_LIMPOS --simular
python -m src.selecionar_imagens --origem DADOS_LIMPOS --saida DADOS_SELECIONADOS --n-teste 40 --n-reserva 60 --lado-maior 1024 --seed 42
```

O primeiro comando apenas simula; revisar o relatório antes de executá-lo sem `--simular`. A seleção exige fotos já gravadas na pasta de origem. Considere sessões/locais antes de definir treino e avaliação. Consulte [FLUXO_DE_DADOS.md](docs/FLUXO_DE_DADOS.md) para o processo real de SAM/CVAT, as etapas externas e as limitações de reprodução sem os dados privados.

## Testes

```bash
python -m unittest discover -s tests -v
```

A suíte usa imagens sintéticas, stubs de modelo e objetos simulados do Telegram: não precisa de pesos, rede, token real ou GPU. O teste opcional relacionado a pi_heif é ignorado quando o pacote não existe. Aprovação da suíte não comprova operação ao vivo do bot.

## Limitações e perspectivas

- Rótulos SAM/CVAT com 1768 avisos geométricos remanescentes em 358 imagens; a triagem não certifica cada contorno. Fonte: chaves `qualidade_rotulos.*`.
- Um local/sessão no teste, sem confirmação em voo independente. O bootstrap por imagem não modela essa diversidade ausente.
- IoU/Dice por união de áreas diferem de mAP de instâncias e não validam medida de terreno em metros quadrados.
- Teste vivo do bot, consentimento das fotos e validação externa permanecem pendentes de comprovação.
- Próximos passos: revisar referências problemáticas, obter fotos de novo local/voo com autorização e registrar validação humana. Qualquer configuração deve ser decidida com val; test serve para confirmação.

## Licenças e créditos

Consulte [LICENCAS_E_CREDITOS.md](docs/LICENCAS_E_CREDITOS.md). Licenças de dependências, dados e publicação dos pesos precisam de avaliação do responsável; não foi criada uma licença global para este repositório.

## Equipe

- Gustavo Barros Martins.
- Hugo Alexandre Carvalho Coelho Coutinho.
- Jeferson Machado dos Santos.
- Maicon de Sousa Pontes.

[PREENCHER: divisão de tarefas confirmada].

Papel do repositório Geo_Map: [PREENCHER: confirmar relação com este projeto].
