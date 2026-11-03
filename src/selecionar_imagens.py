"""Seleciona fotos próprias para teste e reserva, sem alterar os originais.

Uso: python -m src.selecionar_imagens --origem PASTA --saida PASTA_SAIDA
Busca JPG/JPEG/PNG recursivamente apenas dentro da origem informada.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image

from src import config


def data_captura(caminho):
    """Lê DateTimeOriginal/DateTime do EXIF; sem data utilizável, usa mtime."""
    with Image.open(caminho) as imagem:
        exif = imagem.getexif()
        original = exif.get_ifd(34665).get(36867) if 34665 in exif else None
        for texto in (original, exif.get(36867), exif.get(306)):
            if texto:
                try:
                    return datetime.strptime(str(texto)[:19], "%Y:%m:%d %H:%M:%S"), "EXIF"
                except ValueError:
                    continue
    return datetime.fromtimestamp(caminho.stat().st_mtime), "mtime"


def descrever_imagem(caminho):
    """Mede nitidez, dHash de 64 bits e cor média em grade 8x8."""
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
    mini = cv2.resize(cinza, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (mini[:, 1:] > mini[:, :-1]).ravel()
    hash_imagem = sum(int(bit) << i for i, bit in enumerate(bits))
    cor = cv2.resize(imagem, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    caracteristicas = np.concatenate((cor / math.sqrt(192), bits.astype(np.float32) / 8))
    data, fonte = data_captura(caminho)
    return {"caminho": caminho, "data": data, "fonte_data": fonte,
            "nitidez": nitidez, "hash": hash_imagem, "vetor": caracteristicas}


def agrupar_voos(registros):
    """Define novo voo quando o intervalo entre capturas supera dez minutos."""
    anterior, voo = None, 0
    for item in sorted(registros, key=lambda r: (r["data"], str(r["caminho"]))):
        if anterior is None or (item["data"] - anterior).total_seconds() > 600:
            voo += 1
        item["voo"] = voo
        anterior = item["data"]
    return voo


def filtrar_imagens(registros, limiar):
    """Descarta os 10% menos nítidos e quase-duplicatas, mantendo a mais nítida."""
    ordenados = sorted(registros, key=lambda r: (-r["nitidez"], str(r["caminho"])))
    borradas = int(len(ordenados) * 0.10)
    candidatos = ordenados[:len(ordenados) - borradas]
    unicos = []
    for item in candidatos:
        if not any((item["hash"] ^ outro["hash"]).bit_count() <= limiar for outro in unicos):
            unicos.append(item)
    return unicos, borradas, len(candidatos) - len(unicos)


def amostrar_diversas(registros, quantidade, seed):
    """Amostra por ponto mais distante, alternando os voos com imagens disponíveis."""
    if not registros or quantidade <= 0:
        return []
    registros = sorted(registros, key=lambda r: str(r["caminho"]))
    vetores = np.stack([r["vetor"] for r in registros])
    grupos = defaultdict(list)
    for i, item in enumerate(registros):
        grupos[item["voo"]].append(i)
    rng = random.Random(seed)
    voos = sorted(grupos)
    rng.shuffle(voos)
    distancias = np.full(len(registros), np.inf)
    escolhidos = []
    while len(escolhidos) < min(quantidade, len(registros)):
        for voo in voos:
            indices = grupos[voo]
            if not indices:
                continue
            indice = rng.choice(indices) if not escolhidos else max(indices, key=lambda i: distancias[i])
            escolhidos.append(indice)
            indices.remove(indice)
            distancias = np.minimum(distancias, ((vetores - vetores[indice]) ** 2).sum(axis=1))
            if len(escolhidos) >= min(quantidade, len(registros)):
                break
    return [registros[i] for i in escolhidos]


def dividir_selecao(registros, n_teste, n_reserva, seed):
    """Separa voos inteiros para reserva quando há imagens suficientes para teste."""
    grupos = defaultdict(list)
    for item in registros:
        grupos[item["voo"]].append(item)
    voos = sorted(grupos)
    random.Random(seed).shuffle(voos)
    reservados = set()
    total_reservado = 0
    alvo_teste = min(n_teste, len(registros))
    for voo in voos:
        tamanho = len(grupos[voo])
        if total_reservado < n_reserva and len(registros) - total_reservado - tamanho >= alvo_teste:
            reservados.add(voo)
            total_reservado += tamanho
    teste = amostrar_diversas([r for r in registros if r["voo"] not in reservados], n_teste, seed)
    usados = {r["caminho"] for r in teste}
    voos_teste = {r["voo"] for r in teste}
    outros_voos = [r for r in registros if r["voo"] not in voos_teste and r["caminho"] not in usados]
    reserva = amostrar_diversas(outros_voos, n_reserva, seed + 1)
    usados.update(r["caminho"] for r in reserva)
    if len(reserva) < n_reserva:
        restantes = [r for r in registros if r["caminho"] not in usados]
        reserva += amostrar_diversas(restantes, n_reserva - len(reserva), seed + 2)
    return teste, reserva


def gravar_selecao(origem, saida, teste, reserva, lado_maior):
    """Grava JPEGs sem EXIF e um CSV com os nomes relativos dos originais."""
    indice = 0
    with (saida / "mapeamento.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["nome_original", "nome_novo", "voo", "data", "fonte_data"])
        for pasta, itens in [("teste_para_rotular", teste), ("reserva_treino", reserva)]:
            destino = saida / pasta
            destino.mkdir()
            for item in itens:
                indice += 1
                imagem = cv2.imread(str(item["caminho"]))
                if imagem is None:
                    raise ValueError(f"Não foi possível reler: {item['caminho']}")
                h, w = imagem.shape[:2]
                escala = min(1.0, lado_maior / max(h, w))
                tamanho = (max(1, round(w * escala)), max(1, round(h * escala)))
                imagem = cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)
                nome = f"drone_{indice:04d}.jpg"
                if not cv2.imwrite(str(destino / nome), imagem, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Falha ao gravar {destino / nome}")
                escritor.writerow([item["caminho"].relative_to(origem).as_posix(), f"{pasta}/{nome}",
                                   item["voo"], item["data"].isoformat(), item["fonte_data"]])


def selecionar(origem, saida, n_teste=40, n_reserva=60, lado_maior=1024, limiar_duplicata=6, seed=42):
    """Executa a seleção determinística e devolve os totais registrados no relatório."""
    origem, saida = Path(origem).expanduser().resolve(), Path(saida).expanduser().resolve()
    if not origem.is_dir():
        raise FileNotFoundError(f"Origem não encontrada: {origem}")
    if saida == origem or origem in saida.parents:
        raise ValueError("A saída deve ficar fora da origem para preservar a seleção e os originais.")
    if min(n_teste, n_reserva) < 0 or n_teste + n_reserva == 0 or lado_maior <= 0 or not 0 <= limiar_duplicata <= 64:
        raise ValueError("Quantidades devem ser não negativas, com soma positiva; lado maior > 0; limiar entre 0 e 64.")
    caminhos = sorted(p for p in origem.rglob("*") if p.is_file() and p.suffix.lower() in config.EXTENSOES_IMAGEM)
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem JPG/JPEG/PNG na origem: {origem}")
    if saida.exists() and any(saida.iterdir()):
        raise FileExistsError(f"Saída já contém arquivos: {saida}. Escolha uma pasta vazia; nada foi sobrescrito.")
    registros = []
    for i, caminho in enumerate(caminhos, 1):
        registros.append(descrever_imagem(caminho))
        if i % 50 == 0:
            print(f"{i}/{len(caminhos)} imagens analisadas")
    n_voos = agrupar_voos(registros)
    unicos, borradas, duplicatas = filtrar_imagens(registros, limiar_duplicata)
    teste, reserva = dividir_selecao(unicos, n_teste, n_reserva, seed)
    saida.mkdir(parents=True, exist_ok=True)
    gravar_selecao(origem, saida, teste, reserva, lado_maior)
    comuns = sorted({r["voo"] for r in teste} & {r["voo"] for r in reserva})
    avisos = []
    if n_voos == 1:
        avisos.append("Há somente um voo temporal; teste e reserva não são independentes por voo.")
    if comuns:
        avisos.append(f"Voos compartilhados por insuficiência de outros voos: {comuns}. Manter isso explícito na avaliação.")
    if len(teste) < n_teste or len(reserva) < n_reserva:
        avisos.append("Imagens distintas insuficientes após os filtros; quantidades solicitadas não foram completadas.")
    resumo = {"lidas": len(registros), "borradas": borradas, "duplicatas": duplicatas,
              "voos": n_voos, "restantes": len(unicos), "teste": len(teste), "reserva": len(reserva)}
    linhas = ["# Seleção de imagens próprias", "", f"Semente: {seed}; limiar dHash: {limiar_duplicata} bits.",
              "Busca recursiva restrita à origem informada. Voos são grupos temporais (>10 minutos de intervalo).", ""]
    linhas += [f"- {k}: {v}" for k, v in resumo.items()]
    linhas += ["", f"Datas: {dict(Counter(r['fonte_data'] for r in registros))}.",
               f"Teste por voo: {dict(sorted(Counter(r['voo'] for r in teste).items()))}.",
               f"Reserva por voo: {dict(sorted(Counter(r['voo'] for r in reserva).items()))}.", "",
               "Os 10% com menor variância do Laplaciano foram descartados (arredondamento para baixo).",
               "A filtragem por dHash mantém primeiro as imagens mais nítidas. A diversidade usa cor 8x8x3 e dHash.",
               "JPEG qualidade 95, redimensionamento INTER_AREA sem ampliar, lado maior no máximo " + str(lado_maior) + " px.",
               "A gravação com OpenCV remove EXIF, inclusive GPS; originais preservados.",
               "mtime é uma aproximação quando não há EXIF: conferir os voos antes da rotulagem.", ""]
    linhas += ["AVISO: " + a for a in avisos]
    (saida / "relatorio_selecao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return resumo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origem", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--n-teste", type=int, default=40)
    p.add_argument("--n-reserva", type=int, default=60)
    p.add_argument("--lado-maior", type=int, default=1024)
    p.add_argument("--limiar-duplicata", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    try:
        print(selecionar(**vars(args)))
    except (OSError, ValueError) as exc:
        p.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
