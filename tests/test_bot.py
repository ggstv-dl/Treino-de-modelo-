"""Testes offline do bot: apenas imagens sintéticas, sem pesos reais ou GPU."""
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.util
import inspect
import io
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

# Nenhuma credencial real é carregada por esta suíte.
with patch("dotenv.load_dotenv"):
    from src import bot, config

_REDE = []


def setUpModule():
    """Falha imediatamente caso algum teste tente acessar a rede."""
    for nome in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
        bloqueio = patch(nome, side_effect=AssertionError("Rede proibida nos testes"))
        bloqueio.start()
        _REDE.append(bloqueio)


def tearDownModule():
    for bloqueio in reversed(_REDE):
        bloqueio.stop()


class _ArquivosSinteticos:
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="teste-bot-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.pesos = self.raiz / "stub.pt"
        self.pesos.write_bytes(b"stub; nao e modelo")
        self.jpg = self.raiz / "foto.jpg"
        self.png = self.raiz / "foto.png"
        Image.new("RGB", (128, 128), (120, 150, 80)).save(self.jpg)
        Image.new("RGB", (128, 128), (120, 150, 80)).save(self.png)
        self.resolver = patch.object(bot, "caminho_pesos", return_value=self.pesos)
        self.resolver.start()
        self.addCleanup(self.resolver.stop)
        self.modelo = patch.object(bot, "_analisar", side_effect=self.inferir)
        self.inferencia = self.modelo.start()
        self.addCleanup(self.modelo.stop)

    @staticmethod
    def inferir(caminho, *, out_path, pesos):
        Image.new("RGB", (128, 128), "green").save(out_path)
        return str(out_path), {"construcao": 12.5, "vegetacao": 30.0, "via": 5.25}


class TestProcessamento(_ArquivosSinteticos, unittest.TestCase):
    def test_jpg_valido(self):
        saida, areas = bot.processar_imagem(self.jpg)
        self.assertTrue(Path(saida).is_file())
        self.assertTrue(all(isinstance(v, float) for v in areas.values()))
        self.inferencia.assert_called_once_with(
            str(self.jpg), out_path=str(self.raiz / "foto_anotada.jpg"), pesos=str(self.pesos)
        )

    def test_png_valido(self):
        self.assertTrue(Path(bot.processar_imagem(self.png)[0]).exists())

    def test_corrompida(self):
        self.jpg.write_bytes(b"nao e uma imagem")
        with self.assertRaises(bot.ImagemInvalidaError):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_jpeg_truncado_decodificacao(self):
        # Pillow identifica o cabeçalho e verify passa; load deve rejeitar os pixels.
        self.jpg.write_bytes(self.jpg.read_bytes()[:-60])
        with self.assertRaises(bot.ImagemInvalidaError):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_imagem_vazia(self):
        self.jpg.write_bytes(b"")
        with self.assertRaises(bot.ImagemInvalidaError):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_imagem_pequena(self):
        Image.new("RGB", (8, 128)).save(self.jpg)
        with self.assertRaisesRegex(bot.ImagemInvalidaError, "pequena"):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_tamanho_real_excedido(self):
        with self.jpg.open("r+b") as arquivo:
            arquivo.truncate(bot.LIMITE_BYTES + 1)
        with self.assertRaisesRegex(bot.ImagemInvalidaError, "20 MB"):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_fallback_ultralytics_sem_pi_heif(self):
        erro = ModuleNotFoundError("dependencia opcional ausente", name="pi_heif")
        with patch.object(bot.Image, "open", side_effect=erro):
            with self.assertRaises(bot.ImagemInvalidaError):
                bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_nao_oculta_outro_modulo_ausente(self):
        erro = ModuleNotFoundError("outra dependencia", name="outra")
        with patch.object(bot.Image, "open", side_effect=erro):
            with self.assertRaises(ModuleNotFoundError):
                bot.processar_imagem(self.jpg)

    @unittest.skipUnless(importlib.util.find_spec("pi_heif"), "pi_heif ausente; HEIC não suportado")
    def test_heic_opcional_rejeitado(self):
        imagem = Mock(format="HEIF", size=(128, 128))
        contexto = Mock()
        contexto.__enter__ = Mock(return_value=imagem)
        contexto.__exit__ = Mock(return_value=False)
        with patch.object(bot.Image, "open", return_value=contexto):
            with self.assertRaises(bot.ImagemInvalidaError):
                bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_pesos_ausentes(self):
        self.pesos.unlink()
        with self.assertRaises(bot.ModeloIndisponivelError):
            bot.processar_imagem(self.jpg)
        self.inferencia.assert_not_called()

    def test_falha_inferencia(self):
        self.inferencia.side_effect = RuntimeError("falha simulada")
        with self.assertRaises(RuntimeError):
            bot.processar_imagem(self.jpg)

    def test_saida_nao_produzida(self):
        self.inferencia.side_effect = None
        self.inferencia.return_value = (str(self.raiz / "ausente.jpg"), {})
        with self.assertRaisesRegex(RuntimeError, "não produziu"):
            bot.processar_imagem(self.jpg)

    def test_protege_original(self):
        with self.assertRaises(ValueError):
            bot.processar_imagem(self.jpg, saida=self.jpg)
        self.inferencia.assert_not_called()


class TestLegendas(unittest.TestCase):
    def test_percentuais_e_estimativa(self):
        texto = bot.formatar_legenda({"construcao": 12.5, "via": 2.0})
        self.assertIn("Construção: 12,50%", texto)
        self.assertIn("Via: 2,00%", texto)
        self.assertIn("Estimativa", texto)

    def test_sem_deteccao(self):
        texto = bot.formatar_legenda({})
        self.assertIn("Nenhuma classe", texto)
        self.assertIn("ARQUIVO", texto)
        self.assertIn("Estimativa", texto)

    def test_pior_caso_1024(self):
        areas = {f"{i:04d}_" + "🌳" * 3000: 0.1 for i in range(300)}
        for foto in (False, True):
            texto = bot.formatar_legenda(areas, foto_comprimida=foto)
            self.assertLessEqual(len(texto), 1024)
            self.assertLessEqual(len(texto.encode("utf-16-le")) // 2, 1024)
            self.assertIn("Estimativa", texto)
            self.assertIn("omitidas", texto)

    def test_percentual_invalido(self):
        for valor in (float("nan"), float("inf"), -1, 101):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                bot.formatar_legenda({"via": valor})


class TestHandlers(_ArquivosSinteticos, unittest.IsolatedAsyncioTestCase):
    def mensagem(self, *, documento=False, formato="jpg", dados=None):
        self.pastas = []
        async def baixar(*, custom_path, **kwargs):
            self.pastas.append(Path(custom_path).parent)
            Path(custom_path).write_bytes(
                dados if dados is not None else (self.png if formato == "png" else self.jpg).read_bytes()
            )
        self.arquivo = SimpleNamespace(file_size=2000, download_to_drive=AsyncMock(side_effect=baixar))
        self.anexo = SimpleNamespace(
            file_size=2000, file_name=f"foto.{formato}",
            mime_type="image/png" if formato == "png" else "image/jpeg",
            get_file=AsyncMock(return_value=self.arquivo)
        )
        self.msg = SimpleNamespace(
            photo=[] if documento else [self.anexo], document=self.anexo if documento else None,
            reply_text=AsyncMock(), reply_photo=AsyncMock()
        )
        self.update = SimpleNamespace(effective_message=self.msg)
        return self.update

    def conferir_limpeza(self):
        self.assertTrue(self.pastas)
        self.assertTrue(all(not p.exists() for p in self.pastas))

    async def test_start(self):
        await bot.iniciar(self.mensagem(), None)
        texto = self.msg.reply_text.call_args.args[0]
        self.assertLessEqual(len(texto.splitlines()), 5)
        for classe in ("Construção", "Vegetação", "Via", "estimativa", "ângulos"):
            self.assertIn(classe, texto)

    async def test_ajuda(self):
        await bot.iniciar(self.mensagem(), None)
        self.msg.reply_text.assert_awaited_once()
        self.assertIn("ARQUIVO", self.msg.reply_text.call_args.args[0])

    async def test_foto_valida_thread_timeouts_limpeza(self):
        original = asyncio.to_thread
        with patch.object(bot.asyncio, "to_thread", wraps=original) as thread:
            await bot.receber_imagem(self.mensagem(), None)
            thread.assert_awaited_once()
        self.msg.reply_photo.assert_awaited_once()
        self.assertIn("ARQUIVO", self.msg.reply_photo.call_args.kwargs["caption"])
        for chamada in (self.msg.reply_photo, self.arquivo.download_to_drive):
            self.assertEqual(chamada.call_args.kwargs["write_timeout"], 60)
            self.assertEqual(chamada.call_args.kwargs["read_timeout"], 60)
        self.conferir_limpeza()

    async def test_documento_jpg(self):
        await bot.receber_imagem(self.mensagem(documento=True), None)
        self.msg.reply_photo.assert_awaited_once()
        self.conferir_limpeza()

    async def test_documento_png(self):
        await bot.receber_imagem(self.mensagem(documento=True, formato="png"), None)
        self.msg.reply_photo.assert_awaited_once()
        self.conferir_limpeza()

    async def test_documentos_nao_imagem(self):
        for extensao, mime in (("pdf", "application/pdf"), ("zip", "application/zip"), ("txt", "text/plain")):
            with self.subTest(extensao=extensao):
                self.mensagem(documento=True)
                self.anexo.file_name = f"arquivo.{extensao}"
                self.anexo.mime_type = mime
                await bot.receber_imagem(self.update, None)
                self.msg.reply_text.assert_awaited_once()
                self.anexo.get_file.assert_not_awaited()
                self.inferencia.assert_not_called()

    async def test_documento_heic_nao_aceito(self):
        self.mensagem(documento=True)
        self.anexo.file_name, self.anexo.mime_type = "foto.heic", "image/heic"
        await bot.receber_imagem(self.update, None)
        self.assertIn("JPG ou PNG", self.msg.reply_text.call_args.args[0])
        self.anexo.get_file.assert_not_awaited()

    async def test_corrompida_limpeza(self):
        await bot.receber_imagem(self.mensagem(dados=b"jpg quebrado"), None)
        self.assertIn("JPG ou PNG", self.msg.reply_text.call_args.args[0])
        self.inferencia.assert_not_called()
        self.conferir_limpeza()

    async def test_pequena_limpeza(self):
        Image.new("RGB", (4, 4)).save(self.jpg)
        await bot.receber_imagem(self.mensagem(), None)
        self.assertIn("pequena", self.msg.reply_text.call_args.args[0])
        self.inferencia.assert_not_called()
        self.conferir_limpeza()

    async def test_acima_limite_metadados(self):
        self.mensagem()
        self.anexo.file_size = bot.LIMITE_BYTES + 1
        await bot.receber_imagem(self.update, None)
        self.assertIn("20 MB", self.msg.reply_text.call_args.args[0])
        self.anexo.get_file.assert_not_awaited()

    async def test_acima_limite_arquivo_remoto(self):
        self.mensagem()
        self.arquivo.file_size = bot.LIMITE_BYTES + 1
        await bot.receber_imagem(self.update, None)
        self.arquivo.download_to_drive.assert_not_awaited()
        self.assertIn("20 MB", self.msg.reply_text.call_args.args[0])
        self.msg.reply_photo.assert_not_awaited()
        self.inferencia.assert_not_called()

    async def test_download_falha_e_limpa(self):
        self.mensagem()
        async def falhar(*, custom_path, **kwargs):
            self.pastas.append(Path(custom_path).parent)
            Path(custom_path).write_bytes(b"parcial")
            raise OSError("falha de rede simulada")
        self.arquivo.download_to_drive.side_effect = falhar
        await bot.receber_imagem(self.update, None)
        self.assertIn("baixar ou analisar", self.msg.reply_text.call_args.args[0])
        self.inferencia.assert_not_called()
        self.conferir_limpeza()

    async def test_inferencia_falha_e_limpa(self):
        self.inferencia.side_effect = RuntimeError("falha simulada")
        await bot.receber_imagem(self.mensagem(), None)
        self.assertIn("baixar ou analisar", self.msg.reply_text.call_args.args[0])
        self.msg.reply_photo.assert_not_awaited()
        self.conferir_limpeza()

    async def test_envio_falha_e_limpa(self):
        self.mensagem()
        self.msg.reply_photo.side_effect = OSError("falha simulada de envio")
        await bot.receber_imagem(self.update, None)
        self.msg.reply_text.assert_awaited_once()
        self.conferir_limpeza()

    async def test_modelo_ausente(self):
        self.pesos.unlink()
        await bot.receber_imagem(self.mensagem(), None)
        self.assertIn("modelo ainda não", self.msg.reply_text.call_args.args[0])
        self.anexo.get_file.assert_not_awaited()

    async def test_texto_comum(self):
        await bot.orientar(self.mensagem(), None)
        self.assertIn("drone", self.msg.reply_text.call_args.args[0])
        self.inferencia.assert_not_called()

    async def test_sticker_audio_video(self):
        for tipo in ("sticker", "audio", "video"):
            with self.subTest(tipo=tipo):
                self.mensagem()
                self.msg.photo = []
                self.msg.document = None
                setattr(self.msg, tipo, object())
                await bot.orientar(self.update, None)
                self.assertIn("JPG ou PNG", self.msg.reply_text.call_args.args[0])
                self.inferencia.assert_not_called()

    async def test_sem_deteccao(self):
        def vazio(*args, **kwargs):
            saida, _ = self.inferir(*args, **kwargs)
            return saida, {}
        self.inferencia.side_effect = vazio
        await bot.receber_imagem(self.mensagem(), None)
        self.assertIn("Nenhuma classe", self.msg.reply_photo.call_args.kwargs['caption'])

    async def test_erro_global_responde_sem_url(self):
        self.mensagem()
        token = "123456789:" + "A" * 35
        context = SimpleNamespace(error=RuntimeError(f"https://api.telegram.org/bot{token}/getMe"))
        with self.assertLogs(bot.LOGGER, level="ERROR") as captura:
            await bot.tratar_erro(self.update, context)
        self.assertNotIn(token, '\n'.join(captura.output))
        self.msg.reply_text.assert_awaited_once()


class TestConfiguracao(unittest.TestCase):
    def test_resolucao_centralizada(self):
        with patch.dict(os.environ, {"MODEL_WEIGHTS": ""}):
            self.assertEqual(config.resolver_pesos(), config.RAIZ / "models/mvp_final.pt")
            self.assertEqual(bot.caminho_pesos(), config.resolver_pesos())
        with patch.dict(os.environ, {"MODEL_WEIGHTS": "models/alternativo.pt"}):
            self.assertEqual(config.resolver_pesos(), config.RAIZ / "models/alternativo.pt")
            self.assertEqual(config.resolver_pesos("outro.pt"), config.RAIZ / "outro.pt")

    def test_dotenv_carrega_antes_do_padrao(self):
        with patch.dict(os.environ, {"MODEL_WEIGHTS": ""}):
            def carregar(*args, **kwargs):
                os.environ['MODEL_WEIGHTS'] = 'models/carregado.pt'
            try:
                with patch("dotenv.load_dotenv", side_effect=carregar):
                    importlib.reload(config)
                self.assertEqual(config.PESOS_PADRAO, config.RAIZ / "models/carregado.pt")
            finally:
                os.environ['MODEL_WEIGHTS'] = ''
                with patch("dotenv.load_dotenv"):
                    importlib.reload(config)

    def test_predict_e_evaluate_importam_com_stub(self):
        stub = SimpleNamespace(YOLO=Mock(side_effect=AssertionError("Modelo real proibido")))
        with patch.dict(sys.modules, {"ultralytics": stub}):
            predict = importlib.import_module("src.predict")
            evaluate = importlib.import_module("src.evaluate")
            esperado = config.RAIZ / "models/mvp_final.pt"
            self.assertEqual(inspect.signature(predict.analisar).parameters['pesos'].default, esperado)
            class FimDaLeituraDosArgumentos(Exception):
                pass
            def capturar_padrao(parser):
                self.assertEqual(parser.get_default('pesos'), str(esperado))
                raise FimDaLeituraDosArgumentos()
            with patch.object(evaluate.argparse.ArgumentParser, 'parse_args', capturar_padrao):
                with self.assertRaises(FimDaLeituraDosArgumentos):
                    evaluate.main()
            stub.YOLO.assert_not_called()
        # Não deixar os módulos de produção presos ao stub após este teste.
        sys.modules.pop("src.predict", None)
        sys.modules.pop("src.evaluate", None)


class TestInicializacao(unittest.TestCase):
    def setUp(self):
        self.token = "123456789:" + "A" * 35
        self.env = patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": self.token})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.handlers = logging.getLogger().handlers[:]
        self.level = logging.getLogger().level
        self.addCleanup(self.restaurar_logs)

    def restaurar_logs(self):
        raiz = logging.getLogger()
        for handler in raiz.handlers[:]:
            if handler not in self.handlers:
                handler.close()
        raiz.handlers = self.handlers
        raiz.setLevel(self.level)

    def test_sem_token(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": ""}), redirect_stderr(io.StringIO()) as erro:
            self.assertNotEqual(bot.main(["--verificar"]), 0)
        self.assertIn("Defina TELEGRAM_BOT_TOKEN", erro.getvalue())

    def test_verificar_sucesso_uma_consulta(self):
        request = SimpleNamespace(initialize=AsyncMock(), shutdown=AsyncMock())
        cliente = SimpleNamespace(get_me=AsyncMock(return_value=SimpleNamespace(username="drone_teste_bot")))
        with patch.object(bot, 'HTTPXRequest', return_value=request), patch.object(bot, 'Bot', return_value=cliente), redirect_stdout(io.StringIO()) as saida, redirect_stderr(io.StringIO()) as erro:
            codigo = bot.main(["--verificar"])
        self.assertEqual(codigo, 0)
        self.assertEqual(saida.getvalue(), "Bot @drone_teste_bot alcançado\n")
        self.assertEqual(erro.getvalue(), "")
        cliente.get_me.assert_awaited_once()
        request.shutdown.assert_awaited_once()

    def test_verificar_falha_rede_sem_token_em_saida_ou_logs(self):
        request = SimpleNamespace(initialize=AsyncMock(), shutdown=AsyncMock())
        cliente = SimpleNamespace(get_me=AsyncMock(side_effect=OSError(f"https://api.telegram.org/bot{self.token}/getMe")))
        with patch.object(bot, 'HTTPXRequest', return_value=request), patch.object(bot, 'Bot', return_value=cliente), redirect_stdout(io.StringIO()) as saida, redirect_stderr(io.StringIO()) as erro:
            codigo = bot.main(["--verificar"])
            logging.getLogger('httpx').info('https://api.telegram.org/bot%s/getMe', self.token)
            logging.getLogger('httpcore').warning('URL bot%s', self.token)
            try:
                raise OSError('URL bot' + self.token)
            except OSError:
                logging.getLogger('teste').exception('transporte')
        self.assertNotEqual(codigo, 0)
        self.assertIn('Não foi possível conectar', erro.getvalue())
        self.assertNotIn(self.token, saida.getvalue() + erro.getvalue())
        self.assertGreaterEqual(logging.getLogger('httpx').level, logging.WARNING)
        self.assertGreaterEqual(logging.getLogger('httpcore').level, logging.WARNING)
        request.shutdown.assert_awaited_once()

    def test_polling_apenas_stub_registra_handlers_timeouts(self):
        builder = Mock()
        for nome in ('token', 'concurrent_updates', 'read_timeout', 'write_timeout', 'connect_timeout', 'pool_timeout'):
            getattr(builder, nome).return_value = builder
        app = builder.build.return_value
        with patch.object(bot.Application, 'builder', return_value=builder), redirect_stderr(io.StringIO()):
            self.assertEqual(bot.main([]), 0)
        builder.read_timeout.assert_called_once_with(60)
        builder.write_timeout.assert_called_once_with(60)
        comandos, imagens, outros = [c.args[0] for c in app.add_handler.call_args_list]
        self.assertEqual(comandos.commands, frozenset({'start', 'ajuda'}))
        self.assertIs(comandos.callback, bot.iniciar)
        self.assertIs(imagens.callback, bot.receber_imagem)
        self.assertIs(outros.callback, bot.orientar)
        app.add_error_handler.assert_called_once_with(bot.tratar_erro)
        app.run_polling.assert_called_once()


if __name__ == '__main__':
    unittest.main()
