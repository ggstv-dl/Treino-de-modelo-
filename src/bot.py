"""Bot assíncrono de segmentação de fotos aéreas; execute ``python -m src.bot``."""
import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import warnings

from PIL import Image, UnidentifiedImageError
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from src import config

LOGGER = logging.getLogger(__name__)
_INFERENCIA_LOCK = threading.Lock()
_NOMES_LEGIVEIS = {"construcao": "Construção", "vegetacao": "Vegetação", "via": "Via"}
LIMITE_BYTES = 20 * 1024 * 1024
LADO_MINIMO = 64
_TIMEOUTS = dict(read_timeout=60, write_timeout=60, connect_timeout=10, pool_timeout=10)
_ORIENTACAO = "Envie uma foto aérea de drone ou um ARQUIVO de imagem JPG ou PNG. Use /ajuda."


class ModeloIndisponivelError(FileNotFoundError):
    """O checkpoint configurado ainda não está disponível."""


class ImagemInvalidaError(ValueError):
    """O arquivo recebido não é uma imagem JPEG ou PNG utilizável."""


def caminho_pesos(pesos=None):
    """Delega a resolução de pesos à configuração compartilhada."""
    return config.resolver_pesos(pesos)


def _tamanho_telegram(texto):
    # A contagem em unidades UTF-16 também limita nomes contendo emojis.
    return len(texto.encode("utf-16-le")) // 2


def formatar_legenda(areas, *, foto_comprimida=False):
    """Formata áreas em português, respeitando o limite de 1024 da legenda."""
    rodape = "Estimativa do modelo: % da imagem, não metros quadrados."
    dica = "Envie como ARQUIVO (JPG/PNG) para preservar a resolução."
    if not areas:
        return f"Nenhuma classe foi detectada nesta imagem.\n{rodape}\n{dica}"
    linhas = ["Áreas identificadas (% da imagem):"]
    finais = [rodape] + ([dica] if foto_comprimida else [])
    omitidos = False
    for nome, percentual in sorted(areas.items()):
        valor = float(percentual)
        if not math.isfinite(valor) or not 0 <= valor <= 100:
            raise ValueError("Percentual de área inválido.")
        rotulo = _NOMES_LEGIVEIS.get(nome, nome.replace("_", " ").capitalize())
        rotulo = rotulo[:60] + ("…" if len(rotulo) > 60 else "")
        linha = f"{rotulo}: {valor:.2f}%".replace(".", ",")
        if _tamanho_telegram("\n".join(linhas + [linha, "Outras classes omitidas."] + finais)) <= 1024:
            linhas.append(linha)
        else:
            omitidos = True
    if omitidos:
        linhas.append("Outras classes omitidas.")
    return "\n".join(linhas + finais)


def _validar_imagem(entrada):
    """Verifica estrutura e decodifica pixels para detectar truncamento antes do modelo."""
    if not 0 < entrada.stat().st_size <= LIMITE_BYTES:
        raise ImagemInvalidaError("A imagem está vazia ou ultrapassa o limite de 20 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(entrada) as imagem:
                if imagem.format not in {"JPEG", "PNG"}:
                    raise ImagemInvalidaError("Envie uma imagem JPG ou PNG.")
                if min(imagem.size) < LADO_MINIMO:
                    raise ImagemInvalidaError("Imagem muito pequena: cada lado deve ter ao menos 64 pixels.")
                imagem.verify()
            with Image.open(entrada) as imagem:
                imagem.load()
    except ModuleNotFoundError as exc:
        # Ultralytics pode substituir Image.open e tentar HEIC após um JPEG inválido.
        if exc.name != "pi_heif":
            raise
        raise ImagemInvalidaError("Não consegui ler esse arquivo. Envie JPG ou PNG válido.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImagemInvalidaError("Não consegui ler esse arquivo. Envie JPG ou PNG válido.") from exc


def _analisar(*args, **kwargs):
    # Importação tardia evita carregar Ultralytics ou alterar Pillow no --verificar.
    from src.predict import analisar
    return analisar(*args, **kwargs)


def processar_imagem(caminho, *, pesos=None, saida=None):
    """Valida JPEG/PNG e retorna (caminho anotado, áreas), com inferência serializada.

    O chamador controla a vida útil dos arquivos; handlers usam TemporaryDirectory.
    """
    checkpoint = caminho_pesos(pesos)
    if not checkpoint.is_file():
        raise ModeloIndisponivelError("O modelo ainda não está disponível.")
    entrada = Path(caminho)
    _validar_imagem(entrada)
    destino = Path(saida) if saida is not None else entrada.with_name(f"{entrada.stem}_anotada.jpg")
    if destino.resolve() == entrada.resolve():
        raise ValueError("A saída deve ser diferente da imagem original.")
    with _INFERENCIA_LOCK:
        resultado, areas = _analisar(str(entrada), out_path=str(destino), pesos=str(checkpoint))
    if not Path(resultado).is_file():
        raise RuntimeError("A inferência não produziu uma imagem de saída.")
    return resultado, areas


def texto_ajuda(nomes):
    """Monta ajuda em cinco linhas com as classes reais de dataset.yaml."""
    classes = ", ".join(_NOMES_LEGIVEIS.get(n, n) for n in nomes)
    return (
        "Analiso fotos aéreas de drone e devolvo imagem anotada e % de área.\n"
        f"Classes: {classes}.\n"
        "Envie uma FOTO ou um ARQUIVO JPG/PNG para preservar a resolução.\n"
        "O resultado é uma estimativa; o modelo foi treinado em fotos aéreas de drone.\n"
        "Fotos de outros ângulos podem apresentar erros. Use /ajuda."
    )


async def iniciar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Responde a /start e /ajuda."""
    mensagem = update.effective_message
    if mensagem is not None:
        await mensagem.reply_text(texto_ajuda(config.carregar_config_dataset()["names"].values()))


async def orientar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Orienta mensagens de texto e anexos não suportados."""
    if update.effective_message is not None:
        await update.effective_message.reply_text(_ORIENTACAO)


def _validar_anexo(anexo, documento=False):
    """Rejeita tamanho e tipo declarados; os bytes ainda serão validados com PIL."""
    if (anexo.file_size or 0) > LIMITE_BYTES:
        raise ImagemInvalidaError("O arquivo ultrapassa o limite de download de 20 MB. Envie uma versão menor.")
    if documento:
        extensao = Path(anexo.file_name or "").suffix.lower()
        mime = anexo.mime_type or ""
        if (extensao and extensao not in {".jpg", ".jpeg", ".png"}) or (
            mime not in {"image/jpeg", "image/png"} and not (
                mime in {"", "application/octet-stream"} and extensao in {".jpg", ".jpeg", ".png"}
            )
        ):
            raise ImagemInvalidaError("Esse tipo de arquivo não é aceito. Envie uma imagem JPG ou PNG.")


async def receber_imagem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Baixa, valida e infere em thread; descarta temporários mesmo em erro."""
    mensagem = update.effective_message
    if mensagem is None:
        return
    try:
        checkpoint = caminho_pesos()
        if not checkpoint.is_file():
            raise ModeloIndisponivelError()
        foto = bool(mensagem.photo)
        anexo = mensagem.photo[-1] if foto else mensagem.document
        if anexo is None:
            await orientar(update, context)
            return
        _validar_anexo(anexo, documento=not foto)
        with tempfile.TemporaryDirectory(prefix="drone-telegram-") as pasta:
            entrada = Path(pasta) / "entrada.jpg"
            saida = Path(pasta) / "anotada.jpg"
            arquivo = await anexo.get_file()
            _validar_anexo(arquivo)
            await arquivo.download_to_drive(custom_path=entrada, **_TIMEOUTS)
            resultado, areas = await asyncio.to_thread(
                processar_imagem, entrada, pesos=checkpoint, saida=saida
            )
            with Path(resultado).open("rb") as imagem:
                await mensagem.reply_photo(
                    photo=imagem, caption=formatar_legenda(areas, foto_comprimida=foto), **_TIMEOUTS
                )
    except ModeloIndisponivelError:
        await mensagem.reply_text("O modelo ainda não está disponível. Tente novamente mais tarde.")
    except ImagemInvalidaError as exc:
        LOGGER.warning("Arquivo recebido rejeitado na validação.")
        await mensagem.reply_text(str(exc))
    except Exception as exc:
        # Exceções de transporte podem conter o token na URL: registrar somente o tipo.
        LOGGER.error("Falha ao baixar, analisar ou responder à imagem (%s).", type(exc).__name__)
        await mensagem.reply_text("Não consegui baixar ou analisar a imagem agora. Tente novamente em instantes.")


async def tratar_erro(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Registra o tipo do erro global e tenta avisar o usuário, sem expor URLs."""
    LOGGER.error("Falha no processamento de uma atualização (%s).", type(context.error).__name__)
    mensagem = getattr(update, "effective_message", None)
    if mensagem is not None:
        try:
            await mensagem.reply_text("Ocorreu uma falha ao atender sua mensagem. Tente novamente.")
        except Exception as exc:
            LOGGER.warning("Não foi possível enviar o aviso (%s).", type(exc).__name__)


class _FormatterSemToken(logging.Formatter):
    """Remove tokens de mensagens e tracebacks de bibliotecas de transporte."""

    def __init__(self, token):
        super().__init__("%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._token = token

    def format(self, record):
        texto = super().format(record)
        if self._token:
            texto = texto.replace(self._token, "[TOKEN OMITIDO]")
        return re.sub(r"(?:bot)?\d{5,}:[A-Za-z0-9_-]{20,}", "[TOKEN OMITIDO]", texto)


def configurar_logs(token):
    """Reduz logs de transporte e aplica redação de credenciais na saída."""
    handler = logging.StreamHandler()
    handler.setFormatter(_FormatterSemToken(token))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def verificar_bot(token):
    """Consulta getMe uma vez, fecha o cliente HTTP e imprime somente o username."""
    request = HTTPXRequest(**_TIMEOUTS)
    try:
        await request.initialize()
        bot = Bot(token, request=request, get_updates_request=request)
        usuario = await bot.get_me()
        if not usuario.username:
            raise ValueError("Bot sem username")
        print(f"Bot @{usuario.username} alcançado")
        return 0
    finally:
        await request.shutdown()


def main(argv=None):
    """Inicia polling ou apenas verifica a conexão, conforme a linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true", help="Consulta o nome do bot e encerra.")
    args = parser.parse_args(argv)
    # config carrega .env antes de resolver os pesos; a credencial só é lida aqui.
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Defina TELEGRAM_BOT_TOKEN no ambiente ou no arquivo .env para iniciar o bot.", file=sys.stderr)
        return 2
    configurar_logs(token)
    try:
        if args.verificar:
            return asyncio.run(verificar_bot(token))
        application = (
            Application.builder().token(token).concurrent_updates(4)
            .read_timeout(60).write_timeout(60).connect_timeout(10).pool_timeout(10).build()
        )
        application.add_handler(CommandHandler(["start", "ajuda"], iniciar))
        application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, receber_imagem))
        application.add_handler(MessageHandler(filters.ALL, orientar))
        application.add_error_handler(tratar_erro)
        LOGGER.info("Iniciando bot de segmentação de fotos aéreas.")
        application.run_polling()
    except Exception as exc:
        LOGGER.error("Não foi possível conectar ou manter o bot (%s).", type(exc).__name__)
        print("Não foi possível conectar ao Telegram. Confira o token e a conexão.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
