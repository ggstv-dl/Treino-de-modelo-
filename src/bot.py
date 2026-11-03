"""Bot assíncrono do Telegram para segmentação de imagens de drone.

Execute com ``python -m src.bot``. Credenciais são carregadas do ambiente e do
arquivo .env na raiz do projeto apenas ao iniciar o aplicativo.
"""
import asyncio
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from src import config
from src.predict import analisar

LOGGER = logging.getLogger(__name__)
_INFERENCIA_LOCK = threading.Lock()
_NOMES_LEGIVEIS = {"construcao": "Construção", "vegetacao": "Vegetação", "via": "Via"}


class ModeloIndisponivelError(FileNotFoundError):
    """O checkpoint configurado ainda não está disponível."""


class ImagemInvalidaError(ValueError):
    """O arquivo recebido não é uma imagem JPEG ou PNG válida."""


def caminho_pesos(pesos=None):
    """Resolve pesos explícitos, MODEL_WEIGHTS ou o padrão do projeto."""
    escolhido = pesos if pesos is not None else os.environ.get("MODEL_WEIGHTS", "").strip()
    caminho = Path(escolhido).expanduser() if escolhido else config.PESOS_PADRAO
    return caminho if caminho.is_absolute() else config.RAIZ / caminho


def formatar_legenda(areas):
    """Formata percentuais de pixels em português sem depender do Telegram."""
    if not areas:
        return "Nenhuma classe foi detectada nesta imagem."
    linhas = ["Áreas identificadas (% da imagem):"]
    for nome, percentual in sorted(areas.items()):
        valor = float(percentual)
        if not math.isfinite(valor) or not 0 <= valor <= 100:
            raise ValueError("Percentual de área inválido.")
        rotulo = _NOMES_LEGIVEIS.get(nome, nome.replace("_", " ").capitalize())
        linhas.append(f"{rotulo}: {valor:.2f}%".replace(".", ","))
    linhas.append("Percentuais relativos à imagem, não a metros quadrados.")
    return "\n".join(linhas)


def processar_imagem(caminho, *, pesos=None, saida=None):
    """Valida JPEG/PNG e executa a inferência síncrona, com pesos explícitos.

    Retorna (caminho_anotado, areas). O chamador controla a vida útil dos arquivos.
    A exclusão mútua evita inferências simultâneas no modelo compartilhado.
    """
    checkpoint = caminho_pesos(pesos)
    if not checkpoint.is_file():
        raise ModeloIndisponivelError("O modelo ainda não está disponível.")
    entrada = Path(caminho)
    try:
        with Image.open(entrada) as imagem:
            if imagem.format not in {"JPEG", "PNG"}:
                raise ImagemInvalidaError("Envie uma imagem JPG ou PNG.")
            imagem.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImagemInvalidaError("Não foi possível ler a imagem JPG ou PNG.") from exc
    destino = Path(saida) if saida is not None else entrada.with_name(f"{entrada.stem}_anotada.jpg")
    if destino.resolve() == entrada.resolve():
        raise ValueError("A saída deve ser diferente da imagem original.")
    with _INFERENCIA_LOCK:
        resultado, areas = analisar(str(entrada), out_path=str(destino), pesos=str(checkpoint))
    if not Path(resultado).is_file():
        raise RuntimeError("A inferência não produziu uma imagem de saída.")
    return resultado, areas


def texto_ajuda(nomes):
    """Monta a ajuda a partir das classes reais do dataset.yaml."""
    classes = ", ".join(_NOMES_LEGIVEIS.get(n, n) for n in nomes)
    return (
        "Envie uma foto de drone para receber a imagem anotada e os percentuais de área.\n"
        "Para preservar a resolução, envie como documento JPG ou PNG.\n"
        f"Classes identificadas: {classes}.\n"
        "Use /ajuda para ver estas instruções novamente."
    )


async def iniciar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = update.effective_message
    if mensagem is not None:
        nomes = config.carregar_config_dataset()["names"].values()
        await mensagem.reply_text(texto_ajuda(nomes))


async def receber_imagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mensagem = update.effective_message
    if mensagem is None:
        return
    checkpoint = caminho_pesos()
    if not checkpoint.is_file():
        await mensagem.reply_text("O modelo ainda não está disponível. Tente novamente mais tarde.")
        return
    anexo = mensagem.photo[-1] if mensagem.photo else mensagem.document
    if anexo is None:
        await mensagem.reply_text("Envie uma foto ou um documento de imagem JPG ou PNG.")
        return
    try:
        with tempfile.TemporaryDirectory(prefix="drone-telegram-") as pasta:
            entrada = Path(pasta) / "entrada.jpg"
            saida = Path(pasta) / "anotada.jpg"
            arquivo = await anexo.get_file()
            await arquivo.download_to_drive(custom_path=entrada)
            resultado, areas = await asyncio.to_thread(
                processar_imagem, entrada, pesos=checkpoint, saida=saida
            )
            with Path(resultado).open("rb") as imagem:
                await mensagem.reply_photo(photo=imagem, caption=formatar_legenda(areas))
    except ModeloIndisponivelError:
        await mensagem.reply_text("O modelo ainda não está disponível. Tente novamente mais tarde.")
    except ImagemInvalidaError:
        LOGGER.warning("Arquivo recebido não é uma imagem JPEG/PNG válida.")
        await mensagem.reply_text("Não consegui ler esse arquivo. Envie uma imagem JPG ou PNG válida.")
    except Exception:
        LOGGER.exception("Falha ao baixar, analisar ou responder à imagem.")
        await mensagem.reply_text("Não consegui analisar a imagem agora. Tente novamente em instantes.")


async def tratar_erro(update: object, context: ContextTypes.DEFAULT_TYPE):
    # Erros de transporte podem incluir URLs; não registrar o texto da exceção.
    LOGGER.error("Falha no processamento de uma atualização (%s).", type(context.error).__name__)


class _FormatterSemToken(logging.Formatter):
    """Remove o token de mensagens e tracebacks de bibliotecas de transporte."""

    def __init__(self, token):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._token = token

    def format(self, record):
        texto = super().format(record)
        if self._token:
            texto = texto.replace(self._token, "[TOKEN OMITIDO]")
        return re.sub(r"(?:bot)?\d{5,}:[A-Za-z0-9_-]{20,}", "[TOKEN OMITIDO]", texto)


def main():
    load_dotenv(config.RAIZ / ".env", override=False)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Defina TELEGRAM_BOT_TOKEN no ambiente ou no arquivo .env para iniciar o bot.", file=sys.stderr)
        return 2
    handler = logging.StreamHandler()
    handler.setFormatter(_FormatterSemToken(token))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        application = Application.builder().token(token).concurrent_updates(4).build()
        application.add_handler(CommandHandler(["start", "ajuda"], iniciar))
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, receber_imagem))
        application.add_error_handler(tratar_erro)
        LOGGER.info("Iniciando bot; pesos configurados: %s", caminho_pesos())
        application.run_polling()
    except Exception as exc:
        LOGGER.error("Não foi possível iniciar ou manter a conexão do bot (%s).", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
