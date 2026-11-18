"""Avaliação por pixel e comparação pareada, sem alterar treino ou configuração.

As médias globais reutilizam src.evaluate e mantêm sobreposições de rótulo.
Somente a matriz de confusão exclui pixels ambíguos entre classes.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = {0: "construcao", 1: "vegetacao", 2: "via"}


def salvar_json(caminho, dados):
    """Grava JSON legível e rejeita números não finitos."""
    Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def matriz_pixel(rotulo, previsao):
    """Retorna matriz (linha=rótulo), ambíguos e total; índice zero é fundo."""
    gt, pred = np.asarray(rotulo, dtype=bool), np.asarray(previsao, dtype=bool)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError("Máscaras devem ter o mesmo shape (classes, altura, largura).")
    n = gt.shape[0] + 1
    ambiguos = (gt.sum(axis=0) > 1) | (pred.sum(axis=0) > 1)
    mapa_gt = np.where(gt.any(axis=0), gt.argmax(axis=0) + 1, 0)
    mapa_pred = np.where(pred.any(axis=0), pred.argmax(axis=0) + 1, 0)
    indices = (mapa_gt * n + mapa_pred)[~ambiguos]
    matriz = np.bincount(indices, minlength=n * n).reshape(n, n)
    return matriz, int(ambiguos.sum()), int(ambiguos.size)


def contagens(rotulo, previsao):
    """Acumula a união de instâncias por classe, preservando o protocolo do projeto."""
    gt, pred = np.asarray(rotulo, dtype=bool), np.asarray(previsao, dtype=bool)
    return ((gt & pred).sum(axis=(1, 2)), pred.sum(axis=(1, 2)), gt.sum(axis=(1, 2)))


def _razao(a, b):
    return float(a / b) if b else None


def _metricas(inter, pred, gt, calcular):
    por_classe = calcular(inter, pred, gt, CLASSES)
    for c, nome in CLASSES.items():
        dados = por_classe.setdefault(nome, {"iou": None, "dice": None})
        dados.update(intersecao=int(inter[c]), previsao=int(pred[c]), rotulo=int(gt[c]),
                     precisao=_razao(inter[c], pred[c]), revocacao=_razao(inter[c], gt[c]))
    return dict(por_classe=por_classe, **{
        media: float(np.mean([m[chave] for m in por_classe.values() if m[chave] is not None]))
        if any(m[chave] is not None for m in por_classe.values()) else 0.0
        for media, chave in (("mIoU", "iou"), ("mDice", "dice"))
    })


def carregar_grupos(caminho, mapa_interno):
    """Anonimiza locais; sanidade_divisao usa a seleção adjacente como associação."""
    if caminho is None:
        return {}
    caminho = Path(caminho)
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    registros = dados.get("imagens", [])
    if not registros:
        selecao = caminho.with_name("selecao.csv")
        if not selecao.is_file():
            raise ValueError("Sem associação imagem/local: esperado imagens no JSON ou selecao.csv adjacente.")
        with selecao.open(encoding="utf-8-sig", newline="") as f:
            registros = list(csv.DictReader(f, delimiter=";"))
    locais = sorted({r["local"] for r in registros})
    # Codificação alfabética também funciona com mais de 26 locais.
    def sufixo(indice):
        texto = ""
        while indice:
            indice, resto = divmod(indice - 1, 26)
            texto = chr(65 + resto) + texto
        return texto
    mapa = {local: "local_" + sufixo(i + 1) for i, local in enumerate(locais)}
    destino = Path(mapa_interno)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["local_original", "local_anonimo"])
        w.writerows(mapa.items())
    return {r.get("nome", r.get("imagem")): mapa[r["local"]] for r in registros}


def avaliar(pesos, imagens, rotulos, saida, imgsz, conf, grupos=None, mapa_interno=None):
    """Avalia todas as imagens sem exclusões silenciosas e grava resultados rastreáveis."""
    import cv2
    from ultralytics import YOLO
    from src.evaluate import carregar_ground_truth, prever_mascaras, calcular_metricas

    pesos, imagens, rotulos, saida = map(Path, (pesos, imagens, rotulos, saida))
    if not pesos.is_file() or not imagens.is_dir() or not rotulos.is_dir():
        raise FileNotFoundError("Pesos ou diretórios de imagens/rótulos ausentes.")
    arquivos = sorted(p for p in imagens.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not arquivos:
        raise ValueError("Nenhuma imagem encontrada.")
    if len({p.stem for p in arquivos}) != len(arquivos):
        raise ValueError("Há nomes de imagem com o mesmo radical; pareamento ambíguo.")
    for p in arquivos:
        if not (rotulos / (p.stem + ".txt")).is_file():
            raise FileNotFoundError("Rótulo ausente para " + p.name)
    locais = carregar_grupos(grupos, mapa_interno or (
        Path.home() / "evidencias_ted/resultados_finais/scripts/mapa_locais_INTERNO.csv"))
    if grupos and any(p.name not in locais for p in arquivos):
        raise ValueError("Imagem sem local no agrupamento.")
    saida.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(pesos))
    acumulado = np.zeros((3, 3), dtype=np.int64)
    por_local, linhas = {}, []
    matriz = np.zeros((4, 4), dtype=np.int64)
    ambiguos = total = 0
    for p in arquivos:
        imagem = cv2.imread(str(p))
        if imagem is None:
            raise ValueError("Imagem ilegível: " + p.name)
        h, w = imagem.shape[:2]
        gt = carregar_ground_truth(rotulos / (p.stem + ".txt"), h, w, 3)
        pred = prever_mascaras(modelo, p, h, w, 3, conf, imgsz)
        cont = np.array(contagens(gt, pred), dtype=np.int64)
        acumulado += cont
        cm, amb, pixels = matriz_pixel(gt, pred)
        matriz += cm
        ambiguos += amb
        total += pixels
        local = locais.get(p.name, "local_desconhecido")
        agregado = por_local.setdefault(local, {"contagens": np.zeros((3, 3), dtype=np.int64), "imagens": 0})
        agregado["contagens"] += cont
        agregado["imagens"] += 1
        met = _metricas(*cont, calcular_metricas)
        relevantes = [met["por_classe"][nome]["iou"] for c, nome in CLASSES.items() if cont[2, c] > .005 * pixels]
        for c, nome in CLASSES.items():
            m = met["por_classe"][nome]
            linhas.append(dict(imagem=p.name, local=local, classe=nome,
                total_pixels=pixels, ambiguos=amb, objetos_gt=int(gt.any()),
                mIoU=met["mIoU"], mDice=met["mDice"],
                mIoU_relevante=float(np.mean(relevantes)) if relevantes else None,
                falta_fundo=int(cm[c + 1, 0]), excesso_fundo=int(cm[0, c + 1]),
                troca_saida=int(cm[c + 1, 1:].sum() - cm[c + 1, c + 1]),
                troca_entrada=int(cm[1:, c + 1].sum() - cm[c + 1, c + 1]), **m))
    resumo = _metricas(*acumulado, calcular_metricas)
    resumo.update(imagens=len(arquivos), imgsz=int(imgsz), conf=float(conf), retina_masks=True,
        sha256_pesos=hashlib.sha256(pesos.read_bytes()).hexdigest(),
        lista_imagens=[p.name for p in arquivos],
        metodo="Pixels acumulados por classe sem fundo; GT mantém sobreposição. Matriz exclui ambiguidades.",
        locais={k: dict(imagens=v["imagens"], **_metricas(*v["contagens"], calcular_metricas)) for k, v in por_local.items()},
        confusao_pixel=dict(classes=["fundo", *CLASSES.values()], contagens=matriz.tolist(),
            normalizada_por_rotulo=np.divide(matriz, matriz.sum(axis=1, keepdims=True),
                out=np.zeros((4, 4), dtype=float), where=matriz.sum(axis=1, keepdims=True) != 0).tolist(),
            ambiguos=ambiguos, total_pixels=total, fracao_ambigua=ambiguos / total))
    salvar_json(saida / "resumo.json", resumo)
    with (saida / "por_imagem.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)
    with (saida / "confusao_pixel.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rotulo", "previsao", "pixels", "fracao_da_linha"])
        for i, nome in enumerate(resumo["confusao_pixel"]["classes"]):
            for j, outro in enumerate(resumo["confusao_pixel"]["classes"]):
                writer.writerow([nome, outro, int(matriz[i, j]), resumo["confusao_pixel"]["normalizada_por_rotulo"][i][j]])
    return resumo


def ler_contagens(pasta):
    """Lê contagens por imagem/classe em ordem determinística, rejeitando pares duplicados."""
    with (Path(pasta) / "por_imagem.csv").open(encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))
    dados = {}
    for r in linhas:
        chave = (r["imagem"], r["classe"])
        if chave in dados:
            raise ValueError("Par imagem/classe duplicado no CSV.")
        dados[chave] = [int(r[k]) for k in ("intersecao", "previsao", "rotulo")]
    nomes = sorted({r["imagem"] for r in linhas})
    if not nomes or any((nome, c) not in dados for nome in nomes for c in CLASSES.values()):
        raise ValueError("CSV vazio ou sem todas as classes por imagem.")
    return nomes, np.array([[dados[(nome, c)] for c in CLASSES.values()] for nome in nomes], dtype=np.int64)


def _medias_globais(cont):
    inter, pred, gt = (cont[..., i] for i in range(3))
    valores = []
    for numerador, denominador in ((inter, pred + gt - inter), (2 * inter, pred + gt)):
        v = np.divide(numerador, denominador, out=np.zeros_like(numerador, dtype=float), where=denominador > 0)
        presentes = (denominador > 0).sum(axis=-1)
        valores.append(np.divide(v.sum(axis=-1), presentes, out=np.zeros_like(presentes, dtype=float), where=presentes > 0))
    return np.stack(valores, axis=-1)


def bootstrap_pareado(a, b, n_boot=2000, seed=42):
    """Reamostra imagens pareadas e recalcula métricas globais a partir dos pixels."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3 or not len(a) or n_boot < 1:
        raise ValueError("Contagens incompatíveis ou número de reamostragens inválido.")
    if not np.array_equal(a[..., 2], b[..., 2]):
        raise ValueError("Rótulos diferem entre os modelos; comparação inválida.")
    indices = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    ca, cb = a[indices].sum(axis=1), b[indices].sum(axis=1)
    ma, mb = _medias_globais(ca), _medias_globais(cb)
    ponto_a, ponto_b = _medias_globais(a.sum(axis=0)), _medias_globais(b.sum(axis=0))
    resultado = dict(n_boot=n_boot, seed=seed, imagens=len(a),
        metodo="Bootstrap pareado de imagens, IC percentil 95%; não mede independência entre voos/locais.", metricas={}, iou_por_classe={})
    for i, nome in enumerate(("mIoU", "mDice")):
        delta = (mb[:, i] - ma[:, i]) * 100
        resultado["metricas"][nome] = dict(
            A=dict(valor=float(ponto_a[i]), ic95=np.quantile(ma[:, i], [.025, .975]).tolist()),
            B=dict(valor=float(ponto_b[i]), ic95=np.quantile(mb[:, i], [.025, .975]).tolist()),
            diferenca_pp=dict(valor=float((ponto_b[i] - ponto_a[i]) * 100), ic95=np.quantile(delta, [.025, .975]).tolist(),
                positivas=int((delta > 0).sum()), fracao_positiva=float((delta > 0).mean())))
    for c, nome in CLASSES.items():
        resultado["iou_por_classe"][nome] = {}
        for chave, dados, ponto in (("A", ca, a.sum(axis=0)), ("B", cb, b.sum(axis=0))):
            den = dados[:, c, 1] + dados[:, c, 2] - dados[:, c, 0]
            v = np.divide(dados[:, c, 0], den, out=np.full(len(den), np.nan), where=den > 0)
            validos = v[np.isfinite(v)]
            u = ponto[c, 1] + ponto[c, 2] - ponto[c, 0]
            resultado["iou_por_classe"][nome][chave] = dict(valor=_razao(ponto[c, 0], u),
                ic95=np.quantile(validos, [.025, .975]).tolist() if len(validos) else None,
                reamostragens_validas=len(validos))
    return resultado


def comparar(a, b, saida, n_boot=2000, seed=42):
    """Compara somente listas idênticas de imagens e rótulos por classe."""
    nomes_a, ca = ler_contagens(a)
    nomes_b, cb = ler_contagens(b)
    if nomes_a != nomes_b:
        raise ValueError("Listas de imagens diferentes; comparação abortada.")
    resultado = bootstrap_pareado(ca, cb, n_boot, seed)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    salvar_json(saida / "comparacao.json", resultado)
    return resultado


def main(argv=None):
    """Interface de linha de comando sem alteração de datasets ou checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    av = comandos.add_parser("avaliar")
    for nome in ("pesos", "imagens", "rotulos", "saida"):
        av.add_argument("--" + nome, required=True, type=Path)
    av.add_argument("--imgsz", type=int, required=True)
    av.add_argument("--conf", type=float, required=True)
    av.add_argument("--grupos", type=Path)
    comp = comandos.add_parser("comparar")
    for nome in ("a", "b", "saida"):
        comp.add_argument("--" + nome, type=Path, required=True)
    comp.add_argument("--n-boot", type=int, default=2000)
    comp.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args(argv))
    comando = args.pop("comando")
    resultado = avaliar(**args) if comando == "avaliar" else comparar(**args)
    print(json.dumps({k: v for k, v in resultado.items() if k in {"mIoU", "mDice", "imagens", "n_boot"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
