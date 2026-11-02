"""Recupera rótulos antigos e incorpora apenas os cinco primeiros recortes CEU Paz.

Grava um dataset novo em runs/ e preserva imagens, JSON, splits e pesos originais.
Os cinco recortes foram declarados completos pelo usuário; não são revisão automática.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preparar_dataset import (
    extrair_poligonos, formatar_linha, ler_class_dict, ler_mascara,
    mascaras_por_classe, montar_regras,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def converter_anotacao(path, image, names):
    data = json.loads(path.read_text())
    with Image.open(image) as im:
        w, h = im.size
    if (data["imageWidth"], data["imageHeight"]) != (w, h):
        raise ValueError(f"Dimensões divergentes: {path}")
    lines = []
    counts = Counter()
    ids = {name: cid for cid, name in names.items()}
    for shape in data["shapes"]:
        if shape["shape_type"] != "polygon" or shape["label"] not in ids:
            raise ValueError(f"Classe/tipo inesperado em {path}")
        points = np.asarray(shape["points"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"Polígono inválido: {path}")
        if not np.isfinite(points).all() or (points < 0).any() or (points > [w, h]).any():
            raise ValueError(f"Coordenadas inválidas: {path}")
        if cv2.contourArea(points.astype(np.float32)) <= 0:
            raise ValueError(f"Área nula: {path}")
        lines.append(formatar_linha(ids[shape["label"]], points, w, h))
        counts[shape["label"]] += 1
    if not lines:
        raise ValueError(f"Sem polígonos: {path}")
    return lines, dict(counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", type=Path, default=Path.home() / "Downloads/semantic_drone_dataset.zip")
    p.add_argument("--saida", type=Path, default=ROOT / "runs/dataset_ceu_paz_5")
    args = p.parse_args()
    output = args.saida.resolve()
    names = {0: "construcao", 1: "vegetacao", 2: "via"}
    ceu_images = sorted((ROOT / "dataset/rotulagem_ceu_paz/images").glob("*.png"))
    if len(ceu_images) < 5:
        raise ValueError("São necessários cinco recortes")
    selected = ceu_images[:5]
    converted = {p: converter_anotacao(p.with_suffix(".json"), p, names) for p in selected}
    old = {split: sorted((ROOT / "dataset/images" / split).glob("*.jpg"))
           for split in ["train", "val", "test"]}
    if {k: len(v) for k, v in old.items()} != {"train": 280, "val": 80, "test": 40}:
        raise ValueError("Splits originais diferem das contagens auditadas")
    source_files = sum(old.values(), []) + ceu_images + sorted(ceu_images[0].parent.glob("*.json"))
    source_files += sorted((ROOT / "dataset/labels/images").glob("*.txt"))
    source_files += sorted((ROOT / "dataset/rotulagem_propria/images").glob("*.JPG"))
    source_files += list((ROOT / "runs/mvp_drone/weights").glob("*.pt"))
    before = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"selecionadas": [p.name for p in selected],
                "excluidas_incompletas": [p.name for p in ceu_images[5:]],
                "completude": "Cinco primeiras declaradas completas pelo usuário; metadados de revisão dos JSON preservados.",
                "zip_origem": str(args.zip), "zip_sha256": sha256(args.zip),
                "conversao_original": {"area_min_fracao": 0.0005, "epsilon_rel": 0.002,
                    "mascara_resize": "INTER_NEAREST para dimensão da imagem já existente",
                    "buracos": "preservados pelo conversor atual; rótulos antigos não disponíveis para igualdade byte a byte"},
                "classes": names, "arquivos": [], "hashes_originais": before}
    for split in old:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as archive:
        members = archive.namelist()
        masks = {Path(n).stem: n for n in members if "/label_images/" in n and n.endswith(".png")}
        class_files = [n for n in members if n.endswith("class_dict.csv")]
        if len(class_files) != 1:
            raise ValueError("Dicionário de classes ambíguo")
        dictionary = output / "class_dict.csv"
        dictionary.write_bytes(archive.read(class_files[0]))
        rules = montar_regras(ler_class_dict(dictionary), names)
        done = 0
        for split, images in old.items():
            for image in images:
                with Image.open(image) as im:
                    w, h = im.size
                mask = cv2.resize(ler_mascara(io.BytesIO(archive.read(masks[image.stem]))),
                                  (w, h), interpolation=cv2.INTER_NEAREST)
                lines = []
                for cid, region in mascaras_por_classe(mask, rules).items():
                    for points in extrair_poligonos(region, 0.0005 * w * h, 0.002):
                        lines.append(formatar_linha(cid, points, w, h))
                if not lines:
                    raise ValueError(f"Rótulo reconstruído vazio: {image}")
                destination = output / "images" / split / image.name
                shutil.copy2(image, destination)
                label = output / "labels" / split / f"{image.stem}.txt"
                label.write_text("\n".join(lines) + "\n")
                manifest["arquivos"].append({"imagem": str(destination.relative_to(output)),
                    "split": split, "origem": "semantic_drone", "poligonos": len(lines),
                    "imagem_sha256": sha256(destination), "rotulo_sha256": sha256(label)})
                done += 1
                if done % 25 == 0:
                    print(f"Rótulos reconstruídos: {done}/400", flush=True)
    for image, (lines, counts) in converted.items():
        destination = output / "images/train" / image.name
        shutil.copy2(image, destination)
        label = output / "labels/train" / f"{image.stem}.txt"
        label.write_text("\n".join(lines) + "\n")
        manifest["arquivos"].append({"imagem": str(destination.relative_to(output)), "split": "train",
            "origem": "ceu_paz_abril_2022", "poligonos": len(lines), "por_classe": counts,
            "imagem_sha256": sha256(destination), "rotulo_sha256": sha256(label)})
    cfg = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test", "names": names}
    (output / "dataset.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in source_files}:
        raise RuntimeError("Arquivos de origem mudaram durante a preparação")
    groups = {}
    for item in manifest["arquivos"]:
        key = item["imagem_sha256"]
        if key in groups:
            raise ValueError(f"Imagem duplicada no novo dataset: {item['imagem']}")
        groups[key] = item["split"]
    manifest["contagens"] = dict(Counter(item["split"] for item in manifest["arquivos"]))
    manifest["originais_preservados"] = True
    (output / "manifesto.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": str(output), "contagens": manifest["contagens"], "selecionadas": manifest["selecionadas"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
