# Resultados numéricos publicados

Fonte: evidências locais da avaliação de 18/11/2026. `numeros_para_o_relatorio.json` preserva valores, unidades, modelo/SHA8, conjunto, split e ponteiros de origem. Somente prefixos pessoais dos caminhos foram substituídos por `<EVIDENCIAS>`, `<DADOS>` e `<PROJETO>`. O responsável com acesso às fontes privadas pode conferir os ponteiros. As fontes completas não são redistribuídas.

- [Resultados por modelo](resultados_por_modelo.md): blocos de conjuntos separados; não comparar datasets diferentes.
- [Mesmo conjunto e IC95](mesmo_conjunto_ic95.md): principal público versus P3 nas mesmas fotos próprias.
- [Precisão e revocação](precisao_revocacao.md), [peso das classes](peso_das_classes.md) e [mIoU por local anônimo](miou_por_local.md).
- [Conferência](conferencia.md): validação dos valores na etapa de resultados.
- [Gráficos sem fotos](../figuras/LEIA-ME.md): somente curvas, heatmaps e barras.

O bootstrap reamostra imagens da sessão disponível; não representa incerteza entre voos independentes. IoU/Dice são concordância com referências SAM/CVAT, não verdade de campo certificada. A análise do test foi diagnóstica, sem ajuste de modelo ou limiar.
