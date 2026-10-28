"""Converte o Semantic Drone Dataset (máscaras PNG) para o formato YOLO-seg do projeto.

Para cada imagem: redimensiona, junta as classes originais nas classes do projeto
(ver MAPEAMENTO), converte as máscaras em polígonos normalizados e divide em
treino/validação/teste (por imagem, com semente fixa).

Passo 1: conferir se o script entende o seu download (não grava nada):
    python -m src.preparar_dataset --imagens CAMINHO/images --mascaras CAMINHO/masks \\
        --class-dict CAMINHO/class_dict.csv --inspecionar

Passo 2: gerar o dataset em dataset/images e dataset/labels:
    python -m src.preparar_dataset --imagens CAMINHO/images --mascaras CAMINHO/masks \\
        --class-dict CAMINHO/class_dict.csv

Aceita máscaras coloridas (RGB, cores do class_dict.csv) ou de índice (1 canal,
valor = posição da classe no class_dict.csv, começando em 0).
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src import config

# Classe do projeto -> classes originais do Semantic Drone Dataset que ela agrupa.
# Os nomes das classes do projeto precisam ser os mesmos do dataset.yaml.
MAPEAMENTO = {
    "construcao": ["roof", "wall"],
    "vegetacao": ["tree", "grass", "vegetation", "bald-tree"],
    "via": ["paved-area"],
}


def normalizar_nome(nome):
    return nome.strip().lower().replace("_", "-").replace(" ", "-")


def ler_class_dict(caminho):
    """Lê o class_dict.csv (colunas name, r, g, b). Devolve [(nome, (r, g, b)), ...] na ordem do arquivo."""
    with open(caminho, newline="", encoding="utf-8-sig") as f:
        linhas = [[c.strip() for c in linha] for linha in csv.reader(f) if linha]
    cabecalho = [c.lower() for c in linhas[0]]
    if cabecalho[:4] != ["name", "r", "g", "b"]:
        raise ValueError(f"Cabeçalho inesperado em {caminho}: {linhas[0]} (esperado: name, r, g, b)")
    return [(normalizar_nome(l[0]), (int(l[1]), int(l[2]), int(l[3]))) for l in linhas[1:]]


def montar_regras(class_dict, nomes_projeto):
    """Para cada classe do projeto (id), descobre as cores e os índices originais que ela agrupa."""
    por_nome = {nome: (i, cor) for i, (nome, cor) in enumerate(class_dict)}
    ids = {nome: i for i, nome in nomes_projeto.items()}
    faltando_no_yaml = [n for n in MAPEAMENTO if n not in ids]
    if faltando_no_yaml:
        raise ValueError(f"Classes do MAPEAMENTO ausentes no dataset.yaml: {faltando_no_yaml}")
    faltando_no_mapa = [n for n in ids if n not in MAPEAMENTO]
    if faltando_no_mapa:
        raise ValueError(f"Classes do dataset.yaml sem mapeamento: {faltando_no_mapa}")

    regras, ausentes = {}, []
    for nome_projeto, originais in MAPEAMENTO.items():
        indices, cores = [], []
        for o in originais:
            if o not in por_nome:
                ausentes.append(o)
                continue
            indices.append(por_nome[o][0])
            cores.append(por_nome[o][1])
        regras[ids[nome_projeto]] = {"nome": nome_projeto, "originais": originais, "indices": indices, "cores": cores}
    if ausentes:
        disponiveis = [n for n, _ in class_dict]
        raise ValueError(f"Classes originais não encontradas no class_dict.csv: {ausentes}\nDisponíveis: {disponiveis}")
    return regras


def ler_mascara(caminho):
    """Lê a máscara: array 2D (índices) ou 3D (RGB)."""
    arr = np.array(Image.open(caminho))
    return arr[:, :, :3] if arr.ndim == 3 else arr


def mascaras_por_classe(arr, regras):
    """Devolve {id_do_projeto: máscara booleana} para um array de máscara (RGB ou índice)."""
    saida = {}
    if arr.ndim == 3:
        r, g, b = (arr[:, :, i].astype(np.uint32) for i in range(3))
        codigo = (r << 16) | (g << 8) | b
        for cid, regra in regras.items():
            alvo = [(rr << 16) | (gg << 8) | bb for rr, gg, bb in regra["cores"]]
            saida[cid] = np.isin(codigo, alvo)
    else:
        for cid, regra in regras.items():
            saida[cid] = np.isin(arr, regra["indices"])
    return saida


def unir_buracos(contorno, buracos):
    """Liga cada buraco ao contorno por uma ponte percorrida nos dois sentidos.

    YOLO-seg aceita apenas uma sequência de pontos por objeto. As pontes permitem
    representar os anéis internos nessa sequência sem preencher seus interiores.
    Processar da esquerda para a direita permite ligar cada buraco por um raio
    horizontal até a borda mais próxima, inclusive a de um buraco já incorporado.
    """
    pontos = contorno.astype(np.float64)
    for buraco in sorted(buracos, key=lambda b: b[:, 0].min()):
        inicio = np.lexsort((buraco[:, 1], buraco[:, 0]))[0]
        anel = np.roll(buraco, -inicio, axis=0)
        x, y = anel[0]
        a, b = pontos, np.roll(pontos, -1, axis=0)
        dy = b[:, 1] - a[:, 1]
        cruza = (np.minimum(a[:, 1], b[:, 1]) <= y) & (y <= np.maximum(a[:, 1], b[:, 1]))
        xs = np.full(len(pontos), -np.inf)
        inclinadas = cruza & (dy != 0)
        xs[inclinadas] = (a[inclinadas, 0] + (y - a[inclinadas, 1])
                          * (b[inclinadas, 0] - a[inclinadas, 0]) / dy[inclinadas])
        horizontais = cruza & (dy == 0) & (np.minimum(a[:, 0], b[:, 0]) <= x)
        xs[horizontais] = np.minimum(x, np.maximum(a[horizontais, 0], b[horizontais, 0]))
        xs[xs > x] = -np.inf
        i = int(np.argmax(xs))
        if not np.isfinite(xs[i]):
            raise ValueError("Buraco fora do contorno externo: não foi possível preservar a região interna.")
        ponte = np.array([[xs[i], y]])
        pontos = np.concatenate((pontos[:i + 1], ponte, anel, anel[:1], ponte, pontos[i + 1:]))
    return pontos


def extrair_poligonos(mascara, area_min_px, epsilon_rel):
    """Preserva buracos e suas ilhas; simplifica apenas regiões sem anéis internos."""
    contornos, hierarquia = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarquia is None:
        return []
    hierarquia = hierarquia[0]
    poligonos = []
    for i, c in enumerate(contornos):
        if hierarquia[i, 3] != -1 or cv2.contourArea(c) < area_min_px:
            continue
        buracos = []
        filho = hierarquia[i, 2]
        while filho != -1:
            # Não simplifica anéis internos: a simplificação pode apagar buracos
            # pequenos ou cruzar a borda externa em regiões estreitas.
            buracos.append(contornos[filho].reshape(-1, 2))
            filho = hierarquia[filho, 0]
        if buracos:
            # Mantém também a borda externa original para garantir que as pontes
            # fiquem dentro da região, mesmo quando o entorno é fino ou côncavo.
            pontos = unir_buracos(c.reshape(-1, 2), buracos)
        else:
            eps = epsilon_rel * cv2.arcLength(c, True)
            pontos = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(pontos) >= 3:
            poligonos.append(pontos)
    return poligonos


def formatar_linha(cid, pontos, w, h):
    """Uma linha do .txt YOLO-seg: id x1 y1 x2 y2 ... (coordenadas normalizadas)."""
    xs = np.clip(pontos[:, 0] / w, 0, 1)
    ys = np.clip(pontos[:, 1] / h, 0, 1)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{cid} {coords}"


def listar_pares(pasta_imagens, pasta_mascaras):
    """Casa imagem e máscara pelo nome (sem extensão). Devolve (pares, imagens_sem_mascara)."""
    mascaras = {p.stem: p for p in Path(pasta_mascaras).iterdir()
                if p.suffix.lower() in config.EXTENSOES_IMAGEM}
    pares, sem_mascara = [], []
    for img in sorted(Path(pasta_imagens).iterdir()):
        if img.suffix.lower() not in config.EXTENSOES_IMAGEM:
            continue
        if img.stem in mascaras:
            pares.append((img, mascaras[img.stem]))
        else:
            sem_mascara.append(img.name)
    return pares, sem_mascara


def tamanho_final(h, w, lado_maior):
    escala = min(1.0, lado_maior / max(h, w))  # nunca amplia
    return max(1, round(w * escala)), max(1, round(h * escala))


def inspecionar(pares, class_dict, regras, n=3):
    print(f"Pares imagem/máscara encontrados: {len(pares)}")
    print(f"Classes no class_dict.csv ({len(class_dict)}): {[nome for nome, _ in class_dict]}\n")
    cor_para_nome = {cor: nome for nome, cor in class_dict}
    for img, msk in pares[:n]:
        arr = ler_mascara(msk)
        print(f"{img.name} <-> {msk.name}: máscara {arr.shape}, dtype {arr.dtype}")
        total = arr.shape[0] * arr.shape[1]
        if arr.ndim == 3:
            cores, cont = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
            print("  modo: RGB (cores)")
            for cor, c in sorted(zip(map(tuple, cores.tolist()), cont), key=lambda x: -x[1])[:10]:
                print(f"    {cor} -> {cor_para_nome.get(cor, '??? (cor fora do class_dict)')}: {100 * c / total:.1f}%")
        else:
            valores, cont = np.unique(arr, return_counts=True)
            print("  modo: índice (1 canal)")
            for v, c in sorted(zip(valores.tolist(), cont), key=lambda x: -x[1])[:10]:
                nome = class_dict[v][0] if v < len(class_dict) else "??? (fora do class_dict)"
                print(f"    valor {v} -> {nome}: {100 * c / total:.1f}%")
    print("\nMapeamento que será aplicado:")
    for cid, r in regras.items():
        print(f"  {cid} {r['nome']:12s} <- {', '.join(r['originais'])}")
    print("\nSe os nomes e as porcentagens fazem sentido, rode de novo sem --inspecionar.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagens", required=True, help="pasta com as imagens originais")
    p.add_argument("--mascaras", required=True, help="pasta com as máscaras (mesmo nome da imagem, .png)")
    p.add_argument("--class-dict", required=True, help="class_dict.csv do dataset")
    p.add_argument("--saida", default=str(config.RAIZ / "dataset"))
    p.add_argument("--lado-maior", type=int, default=1024, help="redimensiona para este lado maior (padrão: %(default)s)")
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=("TREINO", "VAL", "TESTE"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--area-min", type=float, default=0.0005, help="área mínima de um polígono, fração da imagem (padrão: %(default)s)")
    p.add_argument("--epsilon", type=float, default=0.002, help="simplificação das regiões sem buracos, fração do perímetro (padrão: %(default)s)")
    p.add_argument("--manter-vazias", action="store_true", help="mantém imagens sem nenhum polígono (como fundo puro)")
    p.add_argument("--sobrescrever", action="store_true", help="apaga images/ e labels/ da saída antes de gerar")
    p.add_argument("--inspecionar", action="store_true", help="só mostra o que foi entendido, sem gravar nada")
    args = p.parse_args()

    if abs(sum(args.split) - 1) > 1e-6:
        p.error("--split precisa somar 1 (ex.: 0.7 0.2 0.1)")

    cfg = config.carregar_config_dataset()
    class_dict = ler_class_dict(args.class_dict)
    regras = montar_regras(class_dict, cfg["names"])
    pares, sem_mascara = listar_pares(args.imagens, args.mascaras)
    if not pares:
        raise FileNotFoundError("Nenhum par imagem/máscara encontrado. Confira --imagens e --mascaras.")
    if sem_mascara:
        print(f"Aviso: {len(sem_mascara)} imagens sem máscara correspondente (ignoradas).")

    if args.inspecionar:
        inspecionar(pares, class_dict, regras)
        return

    saida = Path(args.saida)
    for sub in ("images", "labels"):
        pasta = saida / sub
        if pasta.exists() and any(pasta.rglob("*.*")):
            if not args.sobrescrever:
                raise FileExistsError(f"{pasta} já tem arquivos. Use --sobrescrever para refazer.")
            shutil.rmtree(pasta)

    random.Random(args.seed).shuffle(pares)
    n = len(pares)
    n_treino = round(n * args.split[0])
    n_val = round(n * args.split[1])
    destino = {}
    for i, par in enumerate(pares):
        destino[par[0]] = "train" if i < n_treino else "val" if i < n_treino + n_val else "test"
    for split in ("train", "val", "test"):
        (saida / "images" / split).mkdir(parents=True, exist_ok=True)
        (saida / "labels" / split).mkdir(parents=True, exist_ok=True)

    pixels_classe = {cid: 0 for cid in regras}
    pixels_total = 0
    poligonos_classe = {cid: 0 for cid in regras}
    gravadas = {"train": 0, "val": 0, "test": 0}
    vazias = 0

    for k, (img_path, msk_path) in enumerate(pares, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Aviso: não foi possível ler {img_path.name}, ignorada.")
            continue
        w, h = tamanho_final(*img.shape[:2], args.lado_maior)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        arr = cv2.resize(ler_mascara(msk_path), (w, h), interpolation=cv2.INTER_NEAREST)  # nearest: não cria classes novas

        linhas = []
        for cid, mascara in mascaras_por_classe(arr, regras).items():
            pixels_classe[cid] += int(mascara.sum())
            for pontos in extrair_poligonos(mascara, args.area_min * w * h, args.epsilon):
                linhas.append(formatar_linha(cid, pontos, w, h))
                poligonos_classe[cid] += 1
        pixels_total += w * h

        if not linhas and not args.manter_vazias:
            vazias += 1
            continue
        split = destino[img_path]
        cv2.imwrite(str(saida / "images" / split / f"{img_path.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (saida / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(linhas) + ("\n" if linhas else ""))
        gravadas[split] += 1
        if k % 25 == 0 or k == n:
            print(f"  {k}/{n} imagens processadas")

    print(f"\nDataset gravado em: {saida}")
    print(f"Imagens: treino {gravadas['train']}, val {gravadas['val']}, teste {gravadas['test']}"
          f" ({vazias} ignoradas por não terem nenhuma das classes)")
    print(f"{'classe':14s} {'polígonos':>10s} {'% dos pixels':>14s}")
    for cid, r in regras.items():
        print(f"{r['nome']:14s} {poligonos_classe[cid]:10d} {100 * pixels_classe[cid] / max(pixels_total, 1):13.1f}%")


if __name__ == "__main__":
    main()
