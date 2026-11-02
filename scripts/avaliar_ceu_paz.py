"""Avalia pesos no teste antigo, nas fotos próprias e nos cinco recortes de treino.

Usa as mesmas métricas semânticas de src.evaluate; os recortes CEU Paz são treino,
portanto suas métricas não representam generalização. Nenhum original é editado.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import carregar_ground_truth, prever_mascaras, calcular_metricas


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pesos", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=ROOT / "runs/dataset_ceu_paz_5")
    p.add_argument("--device", default="0")
    args = p.parse_args()
    if args.saida.exists():
        raise FileExistsError(args.saida)
    names = {0: "construcao", 1: "vegetacao", 2: "via"}
    ceu = sorted((args.dataset / "images/train").glob("CEUPAZ*.png"))
    if len(ceu) != 5:
        raise ValueError("Esperados exatamente cinco recortes CEU Paz")
    cases = {
        "teste_semantic_drone": (sorted((args.dataset / "images/test").glob("*.jpg")), args.dataset / "labels/test"),
        "fotos_proprias": (sorted((ROOT / "dataset/rotulagem_propria/images").glob("*.JPG")), ROOT / "dataset/labels/images"),
        "ceu_paz_treino": (ceu, args.dataset / "labels/train"),
    }
    paths = [args.pesos]
    for images, labels in cases.values():
        if not images:
            raise ValueError("Conjunto de avaliação vazio")
        paths.extend(images)
        paths.extend(labels / f"{image.stem}.txt" for image in images)
    before = {str(path): sha256(path) for path in paths}
    model = YOLO(str(args.pesos))
    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    model.to(device)
    torch.set_num_threads(2)
    result = {"pesos": str(args.pesos), "pesos_sha256": before[str(args.pesos)],
              "device": device, "imgsz": 640, "conf": 0.25, "conjuntos": {},
              "limites": "CEU Paz é conjunto de treino; fotos próprias são diagnóstico de um voo já consultado. Rótulos antigos foram reconstruídos pelo conversor atual."}
    for case, (images, labels) in cases.items():
        start = time.perf_counter()
        inter, pred_pixels, gt_pixels = (np.zeros(3, dtype=np.int64) for _ in range(3))
        per_image = []
        for image in images:
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Imagem ilegível: {image}")
            h, w = img.shape[:2]
            gt = carregar_ground_truth(labels / f"{image.stem}.txt", h, w, 3)
            pred = prever_mascaras(model, image, h, w, 3, 0.25, 640)
            image_inter = np.logical_and(gt, pred).sum(axis=(1, 2))
            image_pred = pred.sum(axis=(1, 2)).astype(np.int64)
            image_gt = gt.sum(axis=(1, 2)).astype(np.int64)
            inter += image_inter
            pred_pixels += image_pred
            gt_pixels += image_gt
            per_image.append({"imagem": image.name, "por_classe": calcular_metricas(image_inter, image_pred, image_gt, names)})
        metrics = calcular_metricas(inter, pred_pixels, gt_pixels, names)
        result["conjuntos"][case] = {"imagens": len(images), "segundos": time.perf_counter() - start,
            "por_classe": metrics, "mIoU": float(np.mean([m["iou"] for m in metrics.values()])),
            "mDice": float(np.mean([m["dice"] for m in metrics.values()])), "por_imagem": per_image}
        print(case, json.dumps({k: v for k, v in result["conjuntos"][case].items() if k != "por_imagem"}), flush=True)
    if before != {str(path): sha256(path) for path in paths}:
        raise RuntimeError("Os arquivos avaliados mudaram durante a avaliação")
    result["originais_preservados"] = True
    result["hashes_sha256"] = before
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
