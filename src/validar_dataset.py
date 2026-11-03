"""Valida imagens e polígonos YOLO-seg sem carregar o modelo.

Uso: python -m src.validar_dataset [--split val test test_proprio]
"""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path

from src import config


def validar_linha(linha, nc):
    """Devolve o id da classe ou explica por que o polígono é inválido."""
    partes = linha.split()
    try:
        classe = int(partes[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("id de classe deve ser inteiro") from exc
    if not 0 <= classe < nc:
        raise ValueError(f"classe {classe} fora do intervalo 0..{nc - 1}")
    coords = partes[1:]
    if len(coords) < 6 or len(coords) % 2:
        raise ValueError("polígono precisa de pelo menos 3 pontos e número par de coordenadas")
    try:
        valores = [float(c) for c in coords]
    except ValueError as exc:
        raise ValueError("coordenada não numérica") from exc
    if any(not math.isfinite(c) or not 0 <= c <= 1 for c in valores):
        raise ValueError("coordenadas devem ser finitas e estar entre 0 e 1")
    return classe


def listar_imagens(pasta):
    """Lista imagens de um split em ordem determinística."""
    return sorted(p for p in pasta.rglob("*") if p.is_file() and p.suffix.lower() in config.EXTENSOES_IMAGEM)


def validar_dataset(cfg, splits=None):
    """Retorna (resumos, erros), incluindo vazamento por nome ou conteúdo."""
    raiz = Path(cfg["path"])
    base = raiz / "images"
    encontrados = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    escolhidos = list(dict.fromkeys(splits if splits is not None else encontrados))
    erros, resumos = [], {}
    if not escolhidos:
        return {}, ["Nenhum split encontrado em dataset/images/."]
    # Mesmo --split test_proprio compara as imagens com os demais splits existentes.
    vistos_nome, vistos_hash = {}, {}
    for split in encontrados:
        for img in listar_imagens(base / split):
            chaves = [(vistos_nome, img.name.casefold()),
                      (vistos_hash, hashlib.sha256(img.read_bytes()).hexdigest())]
            for vistos, chave in chaves:
                anteriores = vistos.setdefault(chave, [])
                for outro, caminho in anteriores:
                    if outro != split and (outro in escolhidos or split in escolhidos):
                        erros.append(f"Imagem em dois splits: {caminho} e {img}")
                anteriores.append((split, img))
    for split in escolhidos:
        pasta_img, pasta_txt = base / split, raiz / "labels" / split
        if not pasta_img.is_dir():
            erros.append(f"Split {split}: pasta de imagens não encontrada; disponíveis: {', '.join(encontrados)}")
            continue
        imagens = listar_imagens(pasta_img)
        rotulos = sorted(p for p in pasta_txt.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
        if not imagens:
            erros.append(f"Split {split}: nenhuma imagem encontrada")
        if not pasta_txt.is_dir():
            erros.append(f"Split {split}: pasta de rótulos não encontrada")
        por_imagem = Counter(p.relative_to(pasta_img).with_suffix("") for p in imagens)
        por_rotulo = {p.relative_to(pasta_txt).with_suffix(""): p for p in rotulos}
        for nome, quantidade in por_imagem.items():
            if quantidade > 1:
                erros.append(f"Split {split}: mais de uma imagem com o mesmo nome-base: {nome}")
            if nome not in por_rotulo:
                erros.append(f"Split {split}: imagem sem .txt homônimo: {nome}")
        for nome in por_rotulo.keys() - por_imagem.keys():
            erros.append(f"Split {split}: .txt sem imagem: {por_rotulo[nome].name}")
        contagens = Counter({c: 0 for c in cfg["names"]})
        vazias = 0
        for nome, txt in por_rotulo.items():
            linhas = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            if nome in por_imagem and not linhas:
                vazias += por_imagem[nome]
            for numero, linha in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
                if not linha.strip():
                    continue
                try:
                    contagens[validar_linha(linha, cfg.get("nc", len(cfg["names"])))] += 1
                except ValueError as exc:
                    erros.append(f"{split}/{txt.name}:{numero}: {exc}")
        resumos[split] = {"imagens": len(imagens), "sem_poligono": vazias,
                          "poligonos": dict(contagens)}
    return resumos, list(dict.fromkeys(erros))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", nargs="+", help="splits a validar; padrão: todos em dataset/images")
    args = p.parse_args()
    cfg = config.carregar_config_dataset()
    resumos, erros = validar_dataset(cfg, args.split)
    for split, resumo in resumos.items():
        print(f"{split}: {resumo['imagens']} imagens; {resumo['sem_poligono']} imagens sem polígono")
        for classe, nome in cfg["names"].items():
            print(f"  {classe} {nome}: {resumo['poligonos'][classe]} polígonos")
    for erro in erros:
        print(f"ERRO: {erro}")
    print(f"Validação: {len(erros)} erro(s).")
    return 1 if erros else 0


if __name__ == "__main__":
    raise SystemExit(main())
