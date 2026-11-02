"""Mede SAM 2.1 Tiny nas cinco primeiras imagens CEU Paz, sem editar os dados.

Executar CPU antes de GPU para comparar as máscaras com os mesmos prompts.
O runtime GPU fica em runs/runtime_sam2_cuda13, separado da instalação atual.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import sysconfig
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runs/runtime_sam2_cuda13"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/teste_sam2_5imagens_2026-11-02")
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes deve ser positivo")
    if not (RUNTIME / "onnxruntime").is_dir():
        parser.error("Instale o runtime isolado conforme docs/teste_sam2_5imagens.md")
    sys.path.insert(0, str(RUNTIME))

    import cv2
    import numpy as np
    import torch  # Carrega libcudart antes do ORT CUDA, inclusive no teste CPU.
    import onnxruntime as ort
    import psutil

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponível; execute fora do ambiente isolado")
        ort.preload_dlls()
    source = Path(sysconfig.get_paths()["purelib"]) / "anylabeling/services/auto_labeling/__base__/sam2.py"
    spec = importlib.util.spec_from_file_location("sam2_instalado", source)
    sam2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam2)
    weights = Path.home() / "xanylabeling_data/models/sam2_hiera_tiny-r20240801"
    encoder = weights / "sam2.1_hiera_tiny.encoder.onnx"
    decoder = weights / "sam2.1_hiera_tiny.decoder.onnx"
    images = sorted((ROOT / "dataset/rotulagem_ceu_paz/images").glob("*.png"))[:5]
    if len(images) != 5:
        raise RuntimeError("Esperadas cinco imagens")
    originals = images + [p.with_suffix(".json") for p in images] + [encoder, decoder, source]
    before = {str(p): digest(p) for p in originals}
    args.output.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output / f"{args.device}_masks"
    masks_dir.mkdir(exist_ok=True)

    # Os prompts emulam cliques: ponto mais interno de cada polígono existente.
    # As anotações não são consideradas referência de qualidade revisada.
    inputs = []
    for path in images:
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        annotation = json.loads(path.with_suffix(".json").read_text())
        prompts = []
        for index, shape in enumerate(annotation["shapes"]):
            if shape["shape_type"] != "polygon":
                continue
            canvas = np.zeros(image.shape[:2], np.uint8)
            points = np.rint(shape["points"]).astype(np.int32)
            cv2.fillPoly(canvas, [points], 1)
            distances = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
            if not distances.any():
                raise ValueError(f"Polígono vazio: {path.name}, índice {index}")
            y, x = np.unravel_index(distances.argmax(), distances.shape)
            prompts.append({"type": "point", "data": [int(x), int(y)], "label": 1,
                            "classe_origem": shape["label"], "indice_origem": index})
        if not prompts:
            raise RuntimeError(f"Sem prompts em {path}")
        inputs.append((path, image, prompts))

    process = psutil.Process()
    peaks = {"rss_bytes": process.memory_info().rss, "gpu_process_mib": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peaks["rss_bytes"] = max(peaks["rss_bytes"], process.memory_info().rss)
            if args.device == "gpu":
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.splitlines():
                    values = [v.strip() for v in line.split(",")]
                    if len(values) == 2 and values[0] == str(os.getpid()) and values[1].isdigit():
                        peaks["gpu_process_mib"] = max(peaks["gpu_process_mib"], int(values[1]))
            stop.wait(0.5)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    model = sam2.SegmentAnything2ONNX(str(encoder), str(decoder), args.device)
    load_s = time.perf_counter() - start
    providers = {"encoder": model.encoder.session.get_providers(),
                 "decoder": model.decoder.session.get_providers()}
    if args.device == "gpu" and any("CUDAExecutionProvider" not in p for p in providers.values()):
        raise RuntimeError(f"Fallback para CPU: {providers}")
    for session in [model.encoder.session, model.decoder.session]:
        session.disable_fallback()
    start = time.perf_counter()
    embedding = model.encode(inputs[0][1])
    model.predict_masks(embedding, [inputs[0][2][0]])
    del embedding
    warmup_s = time.perf_counter() - start
    records = []
    first_masks = {}
    reference_ious = []
    repeated_ious = []
    start = time.perf_counter()
    try:
        for iteration in range(args.passes):
            for path, image, prompts in inputs:
                visit_start = time.perf_counter()
                encode_start = time.perf_counter()
                embedding = model.encode(image)
                encode_s = time.perf_counter() - encode_start
                masks = []
                times = []
                for prompt in prompts:
                    t = time.perf_counter()
                    logits = model.predict_masks(embedding, [prompt])
                    times.append(time.perf_counter() - t)
                    if not np.isfinite(logits).all():
                        raise RuntimeError(f"Máscara não finita em {path.name}")
                    masks.append(np.asarray(logits[0, 0] > 0, dtype=np.bool_))
                del embedding
                masks = np.stack(masks)
                if iteration == 0:
                    first_masks[path.name] = masks
                    np.savez_compressed(masks_dir / f"{path.stem}.npz", masks=masks)
                    if args.device == "gpu":
                        cpu_path = args.output / "cpu_masks" / f"{path.stem}.npz"
                        if cpu_path.exists():
                            with np.load(cpu_path) as f:
                                cpu_masks = f["masks"]
                            for a, b in zip(cpu_masks, masks, strict=True):
                                union = np.logical_or(a, b).sum()
                                reference_ious.append(float(np.logical_and(a, b).sum() / union) if union else 1.0)
                    preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    colors = {"construcao": (60, 120, 255), "vegetacao": (70, 220, 70), "via": (255, 150, 40)}
                    shown = set()
                    for mask, prompt in zip(masks, prompts, strict=True):
                        label = prompt["classe_origem"]
                        if label in shown:
                            continue
                        shown.add(label)
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(preview, contours, -1, colors.get(label, (255, 255, 255)), 3)
                        cv2.circle(preview, tuple(prompt["data"]), 7, (0, 0, 255), -1)
                    cv2.imwrite(str(args.output / f"{args.device}_{path.stem}.jpg"), preview)
                else:
                    for a, b in zip(first_masks[path.name], masks, strict=True):
                        union = np.logical_or(a, b).sum()
                        repeated_ious.append(float(np.logical_and(a, b).sum() / union) if union else 1.0)
                record = {"passagem": iteration + 1, "imagem": path.name, "prompts": len(prompts),
                          "encoder_s": encode_s, "decoder_s": times,
                          "visita_s_inclui_posprocessamento_e_artefatos": time.perf_counter() - visit_start}
                records.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "decoder_s"}), flush=True)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        monitor.join(timeout=7)
    if before != {str(p): digest(p) for p in originals}:
        raise RuntimeError("Originais mudaram durante o teste; não atribuir a mudança automaticamente ao teste")
    clicks = [t for r in records for t in r["decoder_s"]]
    report = {"device": args.device, "ort_version": ort.__version__, "ort_path": ort.__file__,
              "providers": providers, "passes": args.passes, "images": [p.name for p in images],
              "cliques_medidos": len(clicks), "load_s": load_s, "warmup_s": warmup_s,
              "tempo_total_s_inclui_artefatos": elapsed, "encoder_mediana_s": statistics.median(r["encoder_s"] for r in records),
              "decoder_mediana_s": statistics.median(clicks), "decoder_p95_s": float(np.percentile(clicks, 95)),
              "memoria_picos_amostrados": peaks, "iou_medio_cpu_gpu": statistics.mean(reference_ious) if reference_ious else None,
              "iou_minimo_cpu_gpu": min(reference_ious) if reference_ious else None,
              "iou_minimo_repeticoes": min(repeated_ious) if repeated_ious else None,
              "originais_preservados": True, "hashes_sha256": before,
              "prompts": {p.name: prompts for p, _, prompts in inputs}, "records": records,
              "limites": "Teste do encoder/decoder instalado, sem GUI Qt. Prompts de anotações pendentes; não mede qualidade final nem revisão humana. CPU e GPU usam o mesmo ORT 1.27.0."}
    (args.output / f"{args.device}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["records", "prompts", "hashes_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
