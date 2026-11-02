# Teste SAM 2.1 Tiny nas cinco primeiras imagens CEU Paz

Em 02/11/2026, o SAM 2.1 Hiera Tiny foi testado nas cinco primeiras imagens PNG,
em ordem de nome, de `dataset/rotulagem_ceu_paz/images`. O usuário confirmou esse
lote. Os dados e as anotações originais foram preservados por conferência de
SHA-256 antes e depois de cada execução e novamente ao final.

O ambiente original do X-AnyLabeling 4.2.2 tinha **ONNX Runtime 1.31.1 somente
CPU**, com providers Azure e CPU. A disponibilidade de CUDA no PyTorch não
habilitava automaticamente os modelos ONNX do aplicativo.

Foi instalado **ONNX Runtime GPU 1.27.0**, compatível com CUDA 13, somente em
`runs/runtime_sam2_cuda13`. Nenhum pacote da `.venv` original foi substituído.
As execuções CPU e GPU do benchmark usaram o mesmo runtime 1.27.0 e a mesma
implementação SAM2 instalada no X-AnyLabeling. A documentação oficial descreve
as versões CUDA e o carregamento das bibliotecas por meio do PyTorch:
[ONNX Runtime CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Resultados medidos

| Medida | CPU | GPU RTX 4050 | Comparação |
| --- | ---: | ---: | ---: |
| Carregamento das sessões | 0,222 s | 0,360 s | Custo inicial separado |
| Aquecimento, uma imagem e uma consulta | 1,427 s | 0,548 s | Custo inicial separado |
| Encoder, mediana por imagem | 1,328 s | 0,132 s | **10,07× mais rápido** |
| Decoder, mediana por clique | 44,45 ms | 7,97 ms | **5,58× mais rápido** |
| Decoder, percentil 95 | 55,65 ms | 9,20 ms | Mede resposta do modelo |
| Tempo total das três passagens, incluindo artefatos | 36,47 s | 7,18 s | 5,08×; inclui compressão e comparações |
| Pico amostrado de RAM do processo | 2,18 GiB | 1,92 GiB | Não inclui outros aplicativos |
| Pico amostrado de VRAM do processo | — | **2152 MiB / 2,10 GiB** | Amostragem aproximada a cada 0,5 s |

Foram feitas **três passagens nas cinco imagens**, com 119 prompts em cada
passagem: **357 consultas por dispositivo**, além do aquecimento. O encoder e
o decoder GPU confirmaram `CUDAExecutionProvider` ativo; fallback automático
após erro foi desativado. Nenhum erro, saída não finita ou falta de memória
interrompeu as execuções.

Os prompts são cliques positivos calculados no ponto mais interno de cada
polígono existente. Eles reproduzem consultas interativas sem gerar novas
anotações sobre os originais. As classes dos polígonos só identificam a origem
dos prompts; SAM não foi usado para classificá-las automaticamente.

| Imagem | Prompts por passagem |
| --- | ---: |
| `CEUPAZ2022_DJI_0005_x0500_y0400.png` | 20 |
| `CEUPAZ2022_DJI_0017_x1500_y0900.png` | 18 |
| `CEUPAZ2022_DJI_0022_x1800_y1000.png` | 35 |
| `CEUPAZ2022_DJI_0032_x2500_y1700.png` | 13 |
| `CEUPAZ2022_DJI_0043_x2000_y1800.png` | 33 |

As máscaras binárias das passagens repetidas tiveram IoU 1,0 dentro de cada
dispositivo. Entre CPU e GPU, a IoU média foi **99,958%**, e a mínima, **99,436%**.
Essa comparação avalia consistência numérica, não qualidade contra referência
humana revisada.

## Como abrir o X-AnyLabeling com GPU

Feche a instância antiga após salvar suas anotações. Na raiz do projeto:

```bash
.venv/bin/python scripts/xanylabeling_gpu.py \
  --filename dataset/rotulagem_ceu_paz/images
```

Selecione **Segment Anything 2.1 (Tiny)** no painel de modelos. O iniciador
carrega as bibliotecas CUDA/cuDNN, prioriza o runtime GPU isolado e define
`X_ANYLABELING_DEVICE=GPU` apenas para essa execução. O comando `version` do
iniciador foi executado com sucesso e retornou 4.2.2. A interface gráfica não
foi aberta durante o benchmark, preservando a sessão atual do usuário.

Para voltar ao ambiente original, use o iniciador anterior ou
`.venv/bin/xanylabeling`. A configuração global de dispositivo não foi alterada.

## Como reproduzir o teste

Se a pasta do runtime ainda não existir:

```bash
.venv/bin/python -m pip install --no-deps \
  --target runs/runtime_sam2_cuda13 onnxruntime-gpu==1.27.0
```

Na raiz do projeto, execute CPU antes de GPU, para disponibilizar a referência
de comparação numérica:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/testar_sam2_ceu_paz.py --device cpu
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/testar_sam2_ceu_paz.py --device gpu
```

O teste GPU deve ser executado no terminal com acesso real à GPU. Neste ambiente,
esse acesso exigiu execução fora do isolamento. As versões e o código do SAM
instalado devem coincidir para reproduzir a implementação; os hashes constam no
JSON do teste.

O [resumo em JSON](teste_sam2_5imagens_resultados.json) contém tempos, providers,
hashes e limites do teste. Os arquivos brutos, máscaras e visualizações estão em
`runs/teste_sam2_5imagens_2026-11-02/`: `cpu.json`, `gpu.json`, subpastas
`cpu_masks`/`gpu_masks` e cinco JPG por dispositivo. Cada JPG mostra os contornos
de uma consulta por classe presente, com o clique positivo em vermelho.

## Limites e próximo passo

Este teste exercitou o encoder e o decoder usados pelo aplicativo sem a interface
Qt. Ele não comprova a fluidez da interface, a estabilidade durante horas nem o
comportamento do modo “Segmentar tudo”, que pode disparar muito mais consultas.
Os picos de memória amostrados também não são limites máximos garantidos.

Na data do benchmark, as anotações tinham marcações de revisão pendente e não
foram declaradas corretas por esse teste. Posteriormente, o usuário confirmou
os cinco primeiros recortes como completamente rotulados e orientou usar apenas
esses cinco no fine-tuning, excluindo os outros sete. Os metadados dos JSON
permanecem preservados. Nenhum treinamento foi realizado durante este benchmark.
