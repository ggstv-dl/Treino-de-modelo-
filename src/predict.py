"""Inferência: devolve a imagem com máscaras sobrepostas e a % de área por classe.

A função `analisar` é a que será reaproveitada pelo bot do Telegram:

    from src.predict import analisar
    caminho, areas = analisar("foto.jpg")

Uso na linha de comando:
    python -m src.predict foto.jpg
    python -m src.predict foto.jpg --pesos runs/mvp_drone/weights/best.pt --saida resultado.jpg
"""
import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src import config
from src.mascaras import calcular_areas, resolver_sobreposicoes


@lru_cache(maxsize=2)
def carregar_modelo(pesos):
    """Carrega o modelo uma única vez por arquivo de pesos (importante para o bot)."""
    pesos = Path(pesos)
    if not pesos.exists():
        raise FileNotFoundError(f"Pesos não encontrados: {pesos}. Rode o treino primeiro (python -m src.train).")
    return YOLO(str(pesos))


def analisar(img_path, out_path="resultado.jpg", conf=config.CONF, imgsz=config.IMGSZ, pesos=config.PESOS_PADRAO):
    """Roda o modelo em uma imagem.

    Retorna (caminho_da_imagem_anotada, {nome_da_classe: porcentagem_da_area}).
    Preserva buracos e atribui pixels em disputa à instância de maior confiança.
    """
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    modelo = carregar_modelo(str(pesos))
    r = modelo.predict(str(img_path), imgsz=imgsz, conf=conf, retina_masks=True, verbose=False)[0]
    resolver_sobreposicoes(r)
    cv2.imwrite(str(out_path), r.plot())  # imagem anotada (BGR)

    h, w = r.orig_shape
    mascaras = {}  # id da classe -> máscara binária (união de todas as instâncias da classe)
    if r.masks is not None:
        classes = r.boxes.cls.cpu().numpy().astype(int)
        for binaria, c in zip(r.masks.data.cpu().numpy(), classes):
            mascara = mascaras.setdefault(c, np.zeros((h, w), np.uint8))
            mascara |= (binaria > 0.5).astype(np.uint8)

    areas = calcular_areas(mascaras, r.names, h * w)
    return str(out_path), areas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("imagem")
    p.add_argument("--pesos", default=str(config.PESOS_PADRAO))
    p.add_argument("--saida", default="resultado.jpg")
    p.add_argument("--conf", type=float, default=config.CONF)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    args = p.parse_args()

    caminho, areas = analisar(args.imagem, args.saida, args.conf, args.imgsz, args.pesos)
    print("Imagem salva em:", caminho)
    if not areas:
        print("Nenhuma classe detectada.")
    for nome, pct in areas.items():
        print(f"  {nome}: {pct}% da área")


if __name__ == "__main__":
    main()
