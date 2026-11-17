"""Configurações compartilhadas por train.py, evaluate.py e predict.py.

Todos os caminhos são relativos à raiz do repositório, então os scripts
funcionam independentemente da pasta de onde são executados.
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
DATASET_YAML = RAIZ / "dataset.yaml"
RUNS_DIR = RAIZ / "runs"

NOME_EXPERIMENTO = "mvp_drone"
MODELO_BASE = "yolo11s-seg.pt"  # baixa sozinho na 1ª execução; alternativa leve: yolo11n-seg.pt
# Carregar antes de resolver MODEL_WEIGHTS também atende predict e evaluate.
load_dotenv(RAIZ / ".env", override=False)


def resolver_pesos(pesos=None):
    """Resolve pesos explícitos, MODEL_WEIGHTS ou o checkpoint aprovado do MVP."""
    escolhido = pesos if pesos is not None else os.environ.get("MODEL_WEIGHTS", "").strip()
    caminho = Path(escolhido).expanduser() if escolhido else RAIZ / "models" / "mvp_final.pt"
    return caminho if caminho.is_absolute() else RAIZ / caminho


PESOS_PADRAO = resolver_pesos()

IMGSZ = 640
CONF = 0.25
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png"}


def carregar_config_dataset(caminho=DATASET_YAML):
    """Lê o dataset.yaml e padroniza o conteúdo.

    - 'path' vira um caminho absoluto (relativo à raiz do repositório);
    - 'names' vira sempre um dict {id: nome}, mesmo se estiver escrito como lista.
    """
    with open(caminho, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raiz_dataset = Path(cfg["path"])
    if not raiz_dataset.is_absolute():
        raiz_dataset = RAIZ / raiz_dataset
    cfg["path"] = str(raiz_dataset)

    nomes = cfg["names"]
    if isinstance(nomes, list):
        nomes = dict(enumerate(nomes))
    cfg["names"] = {int(i): str(n) for i, n in nomes.items()}
    return cfg
