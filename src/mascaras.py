"""Resolve disputas entre instâncias antes de medir áreas por classe."""
import numpy as np


def resolver_sobreposicoes(resultado):
    """Atribui cada pixel à instância de maior confiança, preservando seus buracos.

    Atualiza máscaras e caixas do resultado para que a visualização use a mesma
    atribuição das métricas. Empates mantêm a primeira detecção do modelo.
    """
    if resultado.masks is None:
        return resultado
    mascaras = resultado.masks.data > 0.5
    ocupados = mascaras.new_zeros(mascaras.shape[1:])
    ordem = resultado.boxes.conf.argsort(descending=True, stable=True)
    for i in ordem.tolist():
        mascaras[i] &= ~ocupados
        ocupados |= mascaras[i]
    visiveis = mascaras.flatten(1).any(1)
    resultado.update(boxes=resultado.boxes.data[visiveis], masks=mascaras[visiveis])
    return resultado


def calcular_areas(mascaras, nomes, total_pixels):
    """Percentuais em centésimos, incluindo o fundo no ajuste de arredondamento."""
    if not mascaras:
        return {}
    ids = sorted(mascaras)
    contagens = [int(mascaras[c].sum()) for c in ids]
    fundo = total_pixels - sum(contagens)
    if fundo < 0:
        raise ValueError("As máscaras por classe ainda se sobrepõem: área total excedida.")
    # Distribui os centésimos restantes pelos maiores restos. Incluir o fundo
    # garante que arredondar três classes nunca produza uma soma de 100,01%.
    numeradores = np.array(contagens + [fundo], dtype=np.int64) * 10000
    centesimos = numeradores // total_pixels
    faltam = 10000 - int(centesimos.sum())
    ordem = np.argsort(-(numeradores % total_pixels), kind="stable")
    centesimos[ordem[:faltam]] += 1
    return {nomes[c]: int(centesimos[i]) / 100 for i, c in enumerate(ids)}
