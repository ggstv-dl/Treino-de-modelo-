# Fluxo real dos dados

O fluxo abaixo descreve o que foi executado. Os nomes de locais são anônimos; fotos e rótulos não são distribuídos. O roteiro completo de pré-anotação/exportação não está neste repositório.

1. **Fotos próprias → limpeza.** `src/limpar_imagens.py` lê subpastas, corrige orientação EXIF, filtra dimensões/exposição/desfoque, remove quase-duplicatas e grava JPEG reduzido sem EXIF/GPS. A simulação permite revisar descartes antes de escrever imagens.
2. **Seleção.** `src/selecionar_imagens.py` usa nitidez, dHash e diversidade de cor, com agrupamento por intervalos de captura. A ausência de EXIF pode levar ao uso de mtime; o agrupamento temporal não comprova voo independente.
3. **Pré-anotação e revisão.** SAM e CVAT foram ferramentas externas. As anotações passaram por pré-anotação e correções/exportações humanas. Notebooks, scripts de pré-anotação/exportação e exportações originais não fazem parte deste repositório. X-AnyLabeling também foi usado no fluxo de anotação; as versões/licenças precisam ser confirmadas.
4. **Reanálise dos rótulos.** `resumo.json` da reanálise registra 450 imagens, 892 polígonos removidos e 0 adicionados. A contagem de remoções não certifica revisão completa de cada contorno. Fonte: `relatorio_reanalise_450.md` e `resumo.json`.
5. **Pacote P3.** Foram selecionadas 365 fotos, com treino/validação/teste de 253/70/42. As 85 fotos dos grupos com achados prioritários foram excluídas. Fontes: `LEIA-ME.md` do pacote e `resumo.json` da reanálise. A divisão mantém local/sessão inteiro: treino em local_A, local_B e local_D; validação em local_C; teste em local_E. As fontes são `selecao.csv`, `sanidade_divisao.json` e o manifesto do pacote. Não há comprovação de independência geográfica ou de novo voo.
6. **Qualidade remanescente.** O pacote conserva 1768 avisos geométricos em 358 imagens. `avisos_geometria.csv` também inclui avisos apenas de área pequena; sua quantidade de linhas não equivale ao total de avisos geométricos. Fontes públicas dos números: `resultados/numeros_para_o_relatorio.json`, chaves `qualidade_rotulos.*`.
7. **Treino e avaliação.** O treino próprio usou um lançador externo ao repositório. `src.train` usa o dataset configurado no YAML do projeto; não é o lançador original do P3. A avaliação por união de pixels usa `src.evaluate`, e `src.analise_erros` acrescenta contagens por imagem, matriz sem pixels ambíguos e bootstrap pareado. Configuração registrada: conf=0.25, imgsz=640. Nenhum limiar/checkpoint foi escolhido pelo test.

## Reprodutibilidade e acesso

Sem os dados privados, os pesos aprovados e as referências de anotação, não é possível reproduzir treino ou métricas do P3. Os testes sintéticos podem ser executados sem esses artefatos. Quem tiver autorização e os mesmos dados pode reproduzir a avaliação pelas interfaces documentadas no README, passando pesos e pastas explicitamente.

Os scripts históricos em `scripts/` e a configuração correspondente tiveram apenas identificadores, caminhos e referências anonimizados. Os dados privados originais não foram renomeados. Para reutilizar esses roteiros, o responsável deve preparar uma cópia autorizada com os nomes anônimos esperados e ajustar os argumentos/caminhos ao seu ambiente. Isso não é um pacote público de dados nem o fluxo principal do P3.

## Dataset público

`src.preparar_dataset` converte máscaras PNG e o dicionário de classes para polígonos YOLO-seg, mapeia classes e divide por imagem. A avaliação pública e a própria pertencem a blocos diferentes; suas métricas não devem ser comparadas diretamente. Obter o Semantic Drone Dataset na fonte oficial/Kaggle e confirmar suas condições; não redistribuir imagens/máscaras nem presumir uso comercial permitido. Consultar LICENCAS_E_CREDITOS.md.
