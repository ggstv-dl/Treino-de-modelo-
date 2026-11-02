"""Executa o fine-tuning CEU Paz com configuração isolada e monitoramento de GPU."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/finetune_ceu_paz_5.yaml")
    p.add_argument("--smoke", action="store_true", help="três épocas com todos os 285 exemplos")
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ["model", "data", "project"]:
        path = Path(cfg[key])
        cfg[key] = str(path if path.is_absolute() else ROOT / path)
    if args.smoke:
        cfg.update(epochs=3, name=cfg["name"] + "_smoke", plots=False)
    weights = Path(cfg.pop("model"))
    before = hashlib.sha256(weights.read_bytes()).hexdigest()
    stop = threading.Event()
    state = {}
    monitor_thread = None

    def start_monitor(trainer):
        nonlocal monitor_thread
        folder = Path(trainer.save_dir)
        state.update(inicio=datetime.now().astimezone().isoformat(), configuracao=cfg,
                     pesos_origem=str(weights), pesos_origem_sha256=before,
                     amp_efetivo=bool(trainer.amp), batch_efetivo=trainer.batch_size,
                     acumulacao_nominal=trainer.accumulate, lotes_por_epoca=len(trainer.train_loader))
        (folder / "execucao.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

        def monitor():
            with (folder / "gpu.csv").open("w") as file:
                file.write("timestamp, utilization_gpu_percent, memory_used_MiB, power_W\n")
                while not stop.is_set():
                    try:
                        r = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=5)
                        file.write(r.stdout)
                        file.flush()
                    except subprocess.TimeoutExpired:
                        pass
                    stop.wait(1)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    model = YOLO(str(weights))
    model.add_callback("on_train_start", start_monitor)
    start = time.perf_counter()
    try:
        model.train(**cfg)
    finally:
        stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=7)
    if before != hashlib.sha256(weights.read_bytes()).hexdigest():
        raise RuntimeError("Pesos de origem mudaram durante o treino")
    state.update(tempo_total_s_inclui_setup_e_avaliacao=time.perf_counter() - start,
                 pesos_origem_preservados=True, saida=str(model.trainer.save_dir),
                 metricas_finais=model.trainer.metrics)
    (Path(model.trainer.save_dir) / "execucao.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
