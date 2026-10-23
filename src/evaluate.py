"""Calcula IoU e Dice por classe, comparando as máscaras previstas com o ground truth.

As métricas são globais: os pixels de todas as imagens do split são acumulados
antes de calcular IoU = |A∩B| / |A∪B| e Dice = 2|A∩B| / (|A|+|B|).

Uso:
    python -m src.evaluate                      # split "test"
    python -m src.evaluate --split val          # se ainda não separou o teste
    python -m src.evaluate --json metricas.json # salva também em arquivo (útil para o relatório)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src import config


def carregar_ground_truth(arquivo_rotulo, h, w, nc):
    """Converte um .txt YOLO-seg em máscaras binárias de shape (nc, h, w)."""
    gt = np.zeros((nc, h, w), np.uint8)
    arquivo_rotulo = Path(arquivo_rotulo)
    if not arquivo_rotulo.exists():
        return gt  # imagem sem objetos
    for linha in arquivo_rotulo.read_text().splitlines():
        partes = linha.split()
        if len(partes) < 7:  # precisa de classe + ao menos 3 pontos
            continue
        c = int(partes[0])
        if not 0 <= c < nc:
            raise ValueError(f"{arquivo_rotulo}: classe {c} fora do intervalo 0..{nc - 1} definido no dataset.yaml")
        pontos = (np.array(partes[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(gt[c], [pontos], 1)
    return gt


def prever_mascaras(modelo, img_path, h, w, nc, conf, imgsz):
    """Roda o modelo e devolve as máscaras previstas, shape (nc, h, w)."""
    pred = np.zeros((nc, h, w), np.uint8)
    r = modelo.predict(str(img_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    if r.masks is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for poligono, c in zip(r.masks.xy, classes):  # polígonos já em coordenadas da imagem original
            if len(poligono) >= 3:
                cv2.fillPoly(pred[c], [poligono.astype(np.int32)], 1)
    return pred


def calcular_metricas(inter, soma_pred, soma_gt, nomes):
    """Calcula IoU e Dice por classe a partir dos pixels acumulados."""
    resultado = {}
    for c, nome in nomes.items():
        uniao = soma_pred[c] + soma_gt[c] - inter[c]
        if uniao == 0:
            continue  # classe ausente no split
        resultado[nome] = {
            "iou": float(inter[c] / uniao),
            "dice": float(2 * inter[c] / (soma_pred[c] + soma_gt[c])),
        }
    return resultado


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pesos", default=str(config.PESOS_PADRAO))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--json", help="caminho para salvar as métricas em JSON")
    args = p.parse_args()

    cfg = config.carregar_config_dataset()
    nomes = cfg["names"]
    nc = len(nomes)

    pasta_imagens = Path(cfg["path"]) / cfg[args.split]
    pasta_rotulos = Path(cfg["path"]) / cfg[args.split].replace("images", "labels")
    if not pasta_imagens.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {pasta_imagens}")
    if not Path(args.pesos).exists():
        raise FileNotFoundError(f"Pesos não encontrados: {args.pesos}")

    modelo = YOLO(args.pesos)
    inter = np.zeros(nc)
    soma_pred = np.zeros(nc)
    soma_gt = np.zeros(nc)

    imagens = sorted(p for p in pasta_imagens.iterdir() if p.suffix.lower() in config.EXTENSOES_IMAGEM)
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem em {pasta_imagens}")

    for img_path in imagens:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Aviso: não foi possível ler {img_path.name}, ignorada.")
            continue
        h, w = img.shape[:2]
        gt = carregar_ground_truth(pasta_rotulos / f"{img_path.stem}.txt", h, w, nc)
        pred = prever_mascaras(modelo, img_path, h, w, nc, args.conf, args.imgsz)

        for c in range(nc):
            inter[c] += np.logical_and(pred[c], gt[c]).sum()
            soma_pred[c] += pred[c].sum()
            soma_gt[c] += gt[c].sum()

    metricas = calcular_metricas(inter, soma_pred, soma_gt, nomes)
    if not metricas:
        print("Nenhuma classe presente no split avaliado.")
        return

    media_iou = float(np.mean([m["iou"] for m in metricas.values()]))
    media_dice = float(np.mean([m["dice"] for m in metricas.values()]))

    print(f"Split: {args.split} ({len(imagens)} imagens)")
    print(f"{'classe':20s} {'IoU':>8s} {'Dice':>8s}")
    for nome, m in metricas.items():
        print(f"{nome:20s} {m['iou']:8.3f} {m['dice']:8.3f}")
    print(f"{'MÉDIA (mIoU/mDice)':20s} {media_iou:8.3f} {media_dice:8.3f}")

    if args.json:
        saida = {"split": args.split, "imagens": len(imagens), "por_classe": metricas,
                 "mIoU": media_iou, "mDice": media_dice}
        Path(args.json).write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Métricas salvas em:", args.json)


if __name__ == "__main__":
    main()
