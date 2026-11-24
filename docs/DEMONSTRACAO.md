# Demonstração do MVP

O bot recebe fotos aéreas de drone e devolve uma imagem com máscaras e percentuais de construção, vegetação e via. O computador que executa o bot precisa permanecer ligado e conectado à internet.

## Preparar outra máquina

Na raiz do repositório, em Linux com Bash, `flock`, `setsid` e Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edite somente o `.env` local e preencha `TELEGRAM_BOT_TOKEN` com a credencial do BotFather. Não use um exemplo preenchido por outra pessoa. `MODEL_WEIGHTS` é opcional; vazio usa `models/mvp_final.pt`. Credenciais já exportadas no ambiente têm precedência sobre o arquivo.

Obtenha os pesos de **[PREENCHER: link dos pesos]**, salve-os em `models/mvp_final.pt` e confira:

```bash
sha256sum models/mvp_final.pt
```

SHA-256 do P3 aprovado: `0189c55604deaafd81458ce9af39b5850bc5132419f0fb63552ab29a09234c69`. Pesos, fotos e rótulos não estão no Git. Verifique autorização e condições de uso antes de obter ou divulgar esses arquivos.

## Verificar e iniciar

```bash
python -m src.bot --verificar
scripts/rodar_bot.sh
```

O primeiro comando consulta o username e termina, sem enviar mensagens. É um diagnóstico opcional: o lançador também faz sua própria verificação antes de abrir o bot. Encerre qualquer execução anterior de `python -m src.bot` antes de usar o lançador; o lock impede duas instâncias do lançador, mas não detecta um processo iniciado diretamente.

O lançador resolve a raiz automaticamente, confere configuração e pesos, usa `/tmp/bot_demo.lock` e grava logs privados em `logs/bot_demo.log`. Reinicia após erro com intervalo de cinco segundos, até cinco falhas consecutivas; o contador é reiniciado quando uma execução dura mais de sessenta segundos. Uma saída normal encerra o lançador. Ctrl+C/SIGTERM encerram o grupo do filho sem reinício. Um token reconhecido no log causa sua remoção e o encerramento do lançador.

Se disponível, `systemd-inhibit` bloqueia a suspensão durante a execução. Esse recurso depende da sessão e permissões do sistema: ensaie antes da apresentação; se ele falhar, corrija o ambiente antes de abrir a demo. Sem a ferramenta, desligue a suspensão manualmente. Não feche a tampa do notebook durante o uso.

As variáveis opcionais `PYTHON`, `ENV_FILE`, `BOT_CMD` e `BOT_LOG` alteram interpretador, arquivo de ambiente, comando e log. Caminhos relativos partem da raiz. `ENV_FILE` é lido como dotenv; não é executado como código shell. `BOT_CMD` é executado por Bash e deve conter apenas um comando de confiança. Nunca coloque o token nele ou na linha de comando. As pré-checagens continuam obrigatórias quando se personaliza o comando.

## O que mostrar

- **[PREENCHER: link do bot (t.me/...)]**.
- **[PREENCHER: link do vídeo]**.
- Abra o bot no Telegram e envie `/start` ou `/ajuda`.
- Envie uma foto autorizada, preferencialmente como **arquivo JPG/PNG**, sem compressão. Também aceita envio como foto.
- Aguarde a imagem anotada e os percentuais. Envie uma imagem por vez na demonstração.
- Use somente fotos autorizadas e previamente conferidas quanto a pessoas, placas e identificação dos locais. Não projete o terminal, `.env`, logs ou mensagens do BotFather.

O bot aceita JPG/PNG, com limite de vinte MiB e lados de pelo menos sessenta e quatro pixels. Arquivos inválidos e pesos indisponíveis recebem orientação. A inferência roda fora do loop assíncrono e é serializada no processo; isso não constitui garantia de disponibilidade ou capacidade para muitos usuários.

## Limites e contingência

Percentuais representam pixels da imagem, não metros quadrados. O resultado depende do cenário e pode omitir ou confundir regiões. As métricas registradas comparam máscaras com rótulos SAM/CVAT que ainda têm avisos geométricos; não são validação de terreno por especialista. O teste do P3 separa local/sessão, sem comprovar voo independente. Veja [MODELO.md](../models/MODELO.md) e [resultados](resultados/).

O ensaio local mede processamento, sem transporte pelo Telegram. Faça um ensaio real na máquina e na conexão da apresentação. Se não responder no tempo previsto, use o vídeo; sem internet, use a cópia offline. Guarde o vídeo e prints autorizados em mídia local. A demonstração local não é uma hospedagem permanente.
