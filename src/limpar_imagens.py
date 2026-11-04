"""Filtra e reduz fotos de drone, preservando os originais.

Uso: python -m src.limpar_imagens --origem PASTA --saida OUTRA_PASTA --simular
Analisa JPG/JPEG/PNG recursivamente. As montagens são relatórios visuais;
--simular não grava os JPEGs individuais planejados no mapeamento.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import math
from multiprocessing import get_context
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

EXTENSOES = {".jpg", ".jpeg", ".png"}
MOTIVOS = ("ilegivel", "muito_pequena", "exposicao", "sem_conteudo", "borrada", "duplicata")
RELATORIOS = {
    "_antes.txt", "execucao.log", "mapeamento.csv", "descartadas.csv",
    "relatorio_limpeza.md", "montagem_descartadas.jpg", "montagem_mantidas.jpg", "verificacao.txt",
}
COLUNAS_MAPA = (
    "nome_novo", "nome_original", "data_captura", "largura_orig", "altura_orig",
    "largura_nova", "altura_nova", "nitidez", "tamanho_orig_bytes", "tamanho_novo_bytes",
)


def inicializar_worker():
    """Evita multiplicar threads do OpenCV dentro dos processos de trabalho."""
    cv2.setNumThreads(1)


def dimensoes_reduzidas(largura, altura, lado):
    """Preserva proporção e nunca amplia imagens menores que o limite."""
    escala = min(1.0, lado / max(largura, altura))
    return max(1, round(largura * escala)), max(1, round(altura * escala))


def reduzir(imagem, lado):
    """Reduz com INTER_AREA, mantendo a imagem se já couber no limite."""
    h, w = imagem.shape[:2]
    tamanho = dimensoes_reduzidas(w, h, lado)
    return cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA) if tamanho != (w, h) else imagem


def ler_data(exif, mtime):
    """Prefere DateTimeOriginal; sem uma data válida, usa mtime no fuso local."""
    original = exif.get(36867)
    if original is None and 34665 in exif:
        original = exif.get_ifd(34665).get(36867)
    if original:
        try:
            return datetime.strptime(str(original)[:19], "%Y:%m:%d %H:%M:%S").isoformat()
        except ValueError:
            pass
    return datetime.fromtimestamp(mtime, ZoneInfo("America/Sao_Paulo")).replace(tzinfo=None).isoformat()


def ler_rgb(caminho):
    """Carrega pixels RGB já orientados e libera os objetos intermediários."""
    with Image.open(caminho) as original:
        orientada = ImageOps.exif_transpose(original)
        try:
            return orientada.convert("RGB")
        finally:
            orientada.close()


def dhash(cinza):
    """Calcula dHash de 64 bits a partir de uma grade de nove por oito pixels."""
    mini = cv2.resize(cinza, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (mini[:, 1:] > mini[:, :-1]).ravel()
    return sum(int(bit) << i for i, bit in enumerate(bits))


def analisar_imagem(tarefa):
    """Analisa uma imagem por vez e devolve só metadados, sem reter pixels."""
    caminho, nome, lado_minimo = tarefa
    caminho = Path(caminho)
    stat = caminho.stat()
    item = {"caminho": str(caminho), "nome_original": nome,
            "tamanho_orig_bytes": stat.st_size, "motivo": "", "detalhe": ""}
    try:
        with Image.open(caminho) as original:
            item["largura_orig"], item["altura_orig"] = original.size
            item["data_captura"] = ler_data(original.getexif(), stat.st_mtime)
        with ler_rgb(caminho) as rgb:
            item["largura_orientada"], item["altura_orientada"] = rgb.size
            if min(rgb.size) < lado_minimo:
                item.update(motivo="muito_pequena", detalhe=f"dimensoes={rgb.width}x{rgb.height}, minimo={lado_minimo}")
                return item
            pixels = np.asarray(rgb)
            cinza = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            del pixels
            media, desvio = cv2.meanStdDev(cinza)
            media, desvio = float(media[0, 0]), float(desvio[0, 0])
            if media < 25 or media > 235:
                item.update(motivo="exposicao", detalhe=f"media={media:.6f}")
                return item
            if desvio < 10:
                item.update(motivo="sem_conteudo", detalhe=f"desvio={desvio:.6f}")
                return item
            # O limite de análise é fixo, independente do --lado-maior de saída.
            cinza = reduzir(cinza, 1024)
            item["nitidez"] = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
            item["hash"] = dhash(cinza)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        item.update(motivo="ilegivel", detalhe=f"{type(exc).__name__}: {exc}")
    return item


def decidir_filtros(itens, limiar_borrao, limiar_duplicata):
    """Aplica desfoque relativo e compara todos os candidatos com as mantidas."""
    candidatas = [r for r in itens if not r["motivo"]]
    mediana = float(np.median([r["nitidez"] for r in candidatas])) if candidatas else 0.0
    limite = limiar_borrao * mediana
    nitidas = []
    for item in candidatas:
        if item["nitidez"] < limite:
            item.update(motivo="borrada", detalhe=f"nitidez={item['nitidez']:.6f}, limiar={limite:.6f}")
        else:
            nitidas.append(item)
    mantidas = []
    for item in sorted(nitidas, key=lambda r: (-r["nitidez"], r["nome_original"])):
        anterior = next((r for r in mantidas if (r["hash"] ^ item["hash"]).bit_count() <= limiar_duplicata), None)
        if anterior is not None:
            distancia = (anterior["hash"] ^ item["hash"]).bit_count()
            item.update(motivo="duplicata", detalhe=f"duplicata de {anterior['nome_original']}; distancia={distancia}")
        else:
            mantidas.append(item)
    mantidas.sort(key=lambda r: (r["data_captura"], r["nome_original"]))
    return mantidas, mediana, limite


def gravar_imagem(tarefa):
    """Relê somente uma imagem selecionada e grava JPEG sem metadados."""
    caminho, destino, lado, qualidade = tarefa
    with ler_rgb(caminho) as rgb:
        pixels = reduzir(np.asarray(rgb), lado)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        ok, dados = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
        if not ok:
            raise OSError(f"Falha ao codificar {Path(caminho).name}")
        # Nunca substituir uma imagem já existente no destino.
        with Path(destino).open("xb") as arquivo:
            arquivo.write(dados.tobytes())
    return Path(destino).stat().st_size


def executar_pool(funcao, tarefas, workers, etapa):
    """Executa tarefas em processos e informa progresso sem transferir imagens."""
    resultados = []
    with ProcessPoolExecutor(max_workers=workers, initializer=inicializar_worker,
                             mp_context=get_context("spawn")) as pool:
        for i, resultado in enumerate(pool.map(funcao, tarefas, chunksize=1), 1):
            resultados.append(resultado)
            if i % 25 == 0 or i == len(tarefas):
                print(f"{etapa}: {i}/{len(tarefas)}", flush=True)
    return resultados


def montar_amostras(itens, destino, seed, por_motivo=False):
    """Cria relatório visual com miniaturas orientadas e motivos de descarte."""
    rng = random.Random(seed)
    amostra = []
    if por_motivo:
        for motivo in MOTIVOS:
            grupo = sorted((r for r in itens if r["motivo"] == motivo), key=lambda r: r["nome_original"])
            amostra.extend(rng.sample(grupo, min(12, len(grupo))))
    else:
        amostra = rng.sample(itens, min(16, len(itens)))
    largura, altura, colunas = 350, 260, 4
    linhas = max(1, math.ceil(len(amostra) / colunas))
    tela = Image.new("RGB", (colunas * largura, linhas * altura), "white")
    desenho = ImageDraw.Draw(tela)
    fonte = ImageFont.load_default(size=16)
    if not amostra:
        desenho.text((15, 15), "Nenhuma imagem nesta categoria.", fill="black", font=fonte)
    for i, item in enumerate(amostra):
        x, y = (i % colunas) * largura, (i // colunas) * altura
        try:
            with ler_rgb(item["caminho"]) as rgb:
                mini = ImageOps.contain(rgb, (largura - 12, altura - 55))
                tela.paste(mini, (x + (largura - mini.width) // 2, y + 4))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            desenho.text((x + 10, y + 70), "Arquivo ilegível", fill="red", font=fonte)
        desenho.text((x + 6, y + altura - 48), item["motivo"] or "mantida", fill="black", font=fonte)
        nome = item["nome_original"]
        desenho.text((x + 6, y + altura - 25), nome if len(nome) <= 38 else "..." + nome[-35:], fill="black", font=fonte)
    tela.save(destino, quality=90)


def resolucao_mediana(itens, chaves):
    """Resume largura e altura separadamente, sem inventar uma imagem mediana."""
    validos = [r for r in itens if all(c in r for c in chaves)]
    if not validos:
        return "não disponível"
    valores = [float(np.median([r[c] for r in validos])) for c in chaves]
    return f"{valores[0]:g} x {valores[1]:g} px"


def escrever_relatorios(saida, itens, mantidas, mediana, limite, parametros):
    """Grava CSVs auditáveis e um resumo; simulação não estima bytes JPEG."""
    with (saida / "mapeamento.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_MAPA, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(mantidas)
    with (saida / "descartadas.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=("nome_original", "motivo", "detalhe"), extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(r for r in itens if r["motivo"])
    contagens = Counter(r["motivo"] for r in itens if r["motivo"])
    antes = sum(r["tamanho_orig_bytes"] for r in itens)
    depois = sum(r.get("tamanho_novo_bytes", 0) for r in mantidas) if not parametros["simular"] else None
    linhas = ["# Limpeza das imagens do drone", "", "Data da execução: " + datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
              "Modo: " + ("simulação; somente relatórios e montagens" if parametros["simular"] else "execução real"), "",
              f"- lidas: {len(itens)}"]
    linhas += [f"- {motivo}: {contagens[motivo]}" for motivo in MOTIVOS]
    linhas += [f"- mantidas: {len(mantidas)}", "", f"Mediana da nitidez: {mediana:.9f}", f"Limite de desfoque: {limite:.9f}",
               "A mediana usa as imagens que passaram por leitura, tamanho, exposição e conteúdo.",
               "A nitidez é calculada com lado maior de até 1024 px, sem ampliar e independentemente do tamanho de saída.", "",
               f"Tamanho total das imagens de entrada: {antes} bytes ({antes / 1e9:.6f} GB)."]
    if depois is None:
        linhas.append("Tamanho depois e redução: não medidos na simulação; nenhum JPEG individual foi gravado.")
    else:
        reducao = 100 * (1 - depois / antes) if antes else 0.0
        linhas.append(f"Tamanho total dos JPEGs finais: {depois} bytes ({depois / 1e9:.6f} GB); redução: {reducao:.3f}%.")
    linhas += ["Os tamanhos acima comparam imagens; o tamanho da pasta, incluindo relatórios, é conferido na verificação final.",
               "", "Resolução mediana antes (largura x altura originais, excluídas ilegíveis): " + resolucao_mediana([r for r in itens if r["motivo"] != "ilegivel"], ("largura_orig", "altura_orig")),
               "Resolução mediana depois (orientada; planejada na simulação): " + resolucao_mediana(mantidas, ("largura_nova", "altura_nova")),
               "", "## Parâmetros", ""]
    linhas += [f"- {chave}: {valor}" for chave, valor in parametros.items()]
    linhas += ["", "Médias de cinza < 25 ou > 235 são exposição; desvio < 10 é sem conteúdo, nessa ordem.",
               "DHash de 64 bits: candidatos ordenados por nitidez são comparados com todas as imagens mantidas; empates usam o nome original.",
               "A ordem do mapeamento é DateTimeOriginal (EXIF) ou mtime, seguida do nome original para desempate.",
               "As montagens mostram até 12 descartadas por motivo e até 16 mantidas, com semente determinística.",
               "JPEGs gravados com OpenCV não contêm EXIF/GPS. Os arquivos originais são apenas lidos.",
               "Na simulação, nomes e dimensões do mapeamento são planejados; tamanho_novo_bytes fica vazio."]
    (saida / "relatorio_limpeza.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return {"lidas": len(itens), **{m: contagens[m] for m in MOTIVOS}, "mantidas": len(mantidas),
            "mediana": mediana, "limite": limite, "bytes_antes": antes, "bytes_depois": depois}


def limpar(origem, saida, lado_maior=1024, qualidade=95, limiar_borrao=0.35,
           limiar_duplicata=5, lado_minimo=640, workers=4, simular=False, seed=42):
    """Executa a limpeza e devolve os totais sem modificar nenhum original."""
    origem, saida = Path(origem).expanduser().resolve(), Path(saida).expanduser().resolve()
    if not origem.is_dir():
        raise FileNotFoundError(f"Origem não encontrada: {origem}")
    if origem == saida or origem in saida.parents or saida in origem.parents:
        raise ValueError("Origem e saída precisam ser pastas separadas, sem relação de ancestralidade.")
    if lado_maior < 1 or lado_minimo < 1 or workers < 1 or not 1 <= qualidade <= 100:
        raise ValueError("Lados e workers devem ser positivos; qualidade precisa estar entre 1 e 100.")
    if not math.isfinite(limiar_borrao) or limiar_borrao < 0 or not 0 <= limiar_duplicata <= 64:
        raise ValueError("Limiar de borrão deve ser finito e não negativo; dHash entre 0 e 64.")
    caminhos = sorted(p for p in origem.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES)
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem JPG/JPEG/PNG em {origem}")
    if saida.exists():
        desconhecidos = [p.name for p in saida.iterdir() if p.name not in RELATORIOS or p.is_symlink() or not p.is_file()]
        if desconhecidos:
            raise FileExistsError(f"Saída contém arquivos que não podem ser sobrescritos: {', '.join(sorted(desconhecidos))}")
    tarefas = [(str(p), p.relative_to(origem).as_posix(), lado_minimo) for p in caminhos]
    itens = executar_pool(analisar_imagem, tarefas, workers, "Análise")
    mantidas, mediana, limite = decidir_filtros(itens, limiar_borrao, limiar_duplicata)
    saida.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(mantidas, 1):
        item["nome_novo"] = f"drone_{i:04d}.jpg"
        item["largura_nova"], item["altura_nova"] = dimensoes_reduzidas(item["largura_orientada"], item["altura_orientada"], lado_maior)
    if not simular and mantidas:
        tarefas = [(r["caminho"], str(saida / r["nome_novo"]), lado_maior, qualidade) for r in mantidas]
        tamanhos = executar_pool(gravar_imagem, tarefas, workers, "Gravação")
        for item, tamanho in zip(mantidas, tamanhos):
            item["tamanho_novo_bytes"] = tamanho
    parametros = {"lado_maior": lado_maior, "qualidade": qualidade, "limiar_borrao": limiar_borrao,
                  "limiar_duplicata": limiar_duplicata, "lado_minimo": lado_minimo,
                  "workers": workers, "simular": simular, "seed": seed}
    resumo = escrever_relatorios(saida, itens, mantidas, mediana, limite, parametros)
    montar_amostras(itens, saida / "montagem_descartadas.jpg", seed, por_motivo=True)
    montar_amostras(mantidas, saida / "montagem_mantidas.jpg", seed)
    return resumo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origem", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--lado-maior", type=int, default=1024)
    p.add_argument("--qualidade", type=int, default=95)
    p.add_argument("--limiar-borrao", type=float, default=0.35)
    p.add_argument("--limiar-duplicata", type=int, default=5)
    p.add_argument("--lado-minimo", type=int, default=640)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--simular", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    try:
        print(limpar(**vars(args)))
    except (OSError, ValueError) as exc:
        p.exit(1, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
