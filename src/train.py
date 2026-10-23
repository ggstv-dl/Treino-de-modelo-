"""Treina o modelo YOLO-seg com o dataset definido em dataset.yaml.

Uso:
    python -m src.train                    # treino completo
    python -m src.train --teste-rapido     # 3 épocas em 20% dos dados, só para validar o pipeline
    python -m src.train --batch 4          # se der erro de memória

Pesos finais: runs/<nome>/weights/best.pt
"""
import argparse

import yaml
from ultralytics import YOLO

from src import config


def criar_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", default=config.MODELO_BASE, help="pesos pré-treinados (padrão: %(default)s)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ, help="se houver GPU sobrando, teste 960")
    p.add_argument("--batch", type=int, default=8, help="reduza para 4 se der erro de memória")
    p.add_argument("--patience", type=int, default=20, help="para sozinho se parar de melhorar")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--nome", default=config.NOME_EXPERIMENTO)
    p.add_argument("--teste-rapido", action="store_true", help="3 épocas em 20%% dos dados (teste de fumaça)")
    return p


def gravar_yaml_resolvido():
    """Grava uma cópia do dataset.yaml com caminho absoluto.

    Evita que o Ultralytics procure o dataset na sua pasta global de datasets
    em vez da raiz do repositório.
    """
    cfg = config.carregar_config_dataset()
    config.RUNS_DIR.mkdir(exist_ok=True)
    destino = config.RUNS_DIR / "dataset_resolvido.yaml"
    destino.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return destino


def main():
    args = criar_parser().parse_args()

    opcoes = {}
    if args.teste_rapido:
        args.epochs = 3
        args.nome = f"{args.nome}_teste_rapido"
        opcoes["fraction"] = 0.2

    model = YOLO(args.modelo)
    model.train(
        data=str(gravar_yaml_resolvido()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        project=str(config.RUNS_DIR),
        name=args.nome,
        **opcoes,
    )
    print(f"Pesos finais: {config.RUNS_DIR / args.nome / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
