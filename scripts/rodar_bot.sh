#!/usr/bin/env bash
# Executa a demonstração a partir da raiz, com credencial fora dos argumentos.
set -u
umask 077
RAIZ=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
cd "$RAIZ" || exit 1
PYTHON=${PYTHON:-.venv/bin/python}
ENV_FILE=${ENV_FILE:-.env}
BOT_LOG=${BOT_LOG:-logs/bot_demo.log}
export ENV_FILE PYTHONDONTWRITEBYTECODE=1
filho=''
espera=''

erro() { printf '%s\n' "$1" >&2; }

conferir_log() {
    # Nunca devolve linhas do log: uma URL pode conter a credencial.
    if [[ -f "$BOT_LOG" ]] && LC_ALL=C grep -Eq '[0-9]{8,10}:[A-Za-z0-9_-]{35}' "$BOT_LOG"; then
        rm -- "$BOT_LOG"
        erro 'Credencial com formato de token detectada: log apagado. Revise a origem antes de executar novamente.'
        return 1
    fi
}

finalizar() {
    local codigo=$?
    trap - EXIT
    if ! conferir_log; then codigo=1; fi
    exit "$codigo"
}

interromper() {
    local codigo=$1
    trap '' INT TERM
    # Cada execução tem seu próprio grupo; encerra também subprocessos do bot.
    if [[ -n "$filho" ]]; then
        kill -TERM -- "-$filho" 2>/dev/null || true
        for ((i=0; i<50; i++)); do
            kill -0 -- "-$filho" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL -- "-$filho" 2>/dev/null || true
        wait "$filho" 2>/dev/null || true
    fi
    if [[ -n "$espera" ]]; then
        kill -TERM "$espera" 2>/dev/null || true
        wait "$espera" 2>/dev/null || true
    fi
    printf '%s\n' 'Demonstração encerrada por sinal; o bot não será reiniciado.'
    exit "$codigo"
}

[[ -f "$ENV_FILE" ]] || { erro 'Arquivo ENV_FILE não encontrado. Copie .env.example para .env e preencha no editor.'; exit 2; }
# Somente a contagem é consultada, sem exibir conteúdo ou token.
quantidade=$(grep -c '^TELEGRAM_BOT_TOKEN=.\+' "$ENV_FILE")
[[ "$quantidade" -gt 0 ]] || { erro 'TELEGRAM_BOT_TOKEN sem valor no ENV_FILE.'; exit 2; }
command -v "$PYTHON" >/dev/null 2>&1 || { erro 'Interpretador PYTHON não encontrado.'; exit 2; }
for ferramenta in flock setsid; do
    command -v "$ferramenta" >/dev/null 2>&1 || { erro "Ferramenta necessária ausente: $ferramenta."; exit 2; }
done
exec 9>/tmp/bot_demo.lock
flock -n 9 || { erro 'Outra instância do lançador já está ativa. Encerre-a antes de abrir outra.'; exit 3; }
mkdir -p -- "$(dirname -- "$BOT_LOG")" || exit 2
touch -- "$BOT_LOG" || exit 2
trap finalizar EXIT
trap 'interromper 130' INT
trap 'interromper 143' TERM
conferir_log || exit 1

# dotenv interpreta o arquivo como dados; nunca o executamos como shell.
# O carregamento usa a mesma precedência de src.config: ambiente prevalece.
carregar_env='import os, sys; from dotenv import load_dotenv; load_dotenv(os.environ["ENV_FILE"], override=False); os.execvp(sys.argv[1], sys.argv[1:])'
"$PYTHON" -c 'import os, sys; from pathlib import Path; from dotenv import load_dotenv
load_dotenv(os.environ["ENV_FILE"], override=False)
if not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
    print("Token vazio após interpretar ENV_FILE.", file=sys.stderr); sys.exit(2)
value = os.environ.get("MODEL_WEIGHTS", "").strip()
path = Path(value).expanduser() if value else Path("models/mvp_final.pt")
if not path.is_file():
    print("Pesos não encontrados; confira MODEL_WEIGHTS ou models/mvp_final.pt.", file=sys.stderr); sys.exit(2)
' >> "$BOT_LOG" 2>&1
codigo=$?
conferir_log || exit 1
if [[ "$codigo" -ne 0 ]]; then
    erro 'Pré-checagem local falhou: confira token, dependências e arquivo dos pesos. O log contém os detalhes sem ser exibido.'
    exit "$codigo"
fi

printf '%s\n' 'Verificando acesso ao Telegram antes de iniciar...'
# Verificação única neste lançamento; não inicia polling.
setsid "$PYTHON" -c "$carregar_env" "$PYTHON" -m src.bot --verificar >> "$BOT_LOG" 2>&1 9>&- &
filho=$!
wait "$filho"; codigo=$?
filho=''
conferir_log || exit 1
if [[ "$codigo" -ne 0 ]]; then
    erro 'Verificação do Telegram falhou. O bot não foi iniciado; confira a conexão e a configuração.'
    exit "$codigo"
fi

inibidor=()
if command -v systemd-inhibit >/dev/null 2>&1; then
    inibidor=(systemd-inhibit --what=sleep --mode=block --who=DroneMVP --why=Demonstracao --)
else
    erro 'systemd-inhibit ausente: desative a suspensão manualmente durante a demonstração.'
fi
if [[ -n "${BOT_CMD:-}" ]]; then
    comando=(bash -c "exec $BOT_CMD")
else
    comando=("$PYTHON" -m src.bot)
fi

falhas=0
while true; do
    printf '%s\n' 'Bot em execução. Use Ctrl+C para encerrar; mensagens detalhadas ficam no log privado.'
    inicio=$SECONDS
    setsid "${inibidor[@]}" "$PYTHON" -c "$carregar_env" "${comando[@]}" >> "$BOT_LOG" 2>&1 9>&- &
    filho=$!
    wait "$filho"; codigo=$?
    filho=''
    duracao=$((SECONDS - inicio))
    conferir_log || exit 1
    if [[ "$codigo" -eq 0 ]]; then
        printf '%s\n' 'Bot encerrado normalmente.'
        exit 0
    fi
    if [[ "$duracao" -gt 60 ]]; then falhas=0; fi
    falhas=$((falhas + 1))
    erro "Bot encerrou com erro; falha consecutiva $falhas de 5 (duração: ${duracao}s)."
    if [[ "$falhas" -ge 5 ]]; then
        erro 'Limite de cinco falhas atingido. Corrija a causa antes de iniciar novamente.'
        exit 1
    fi
    printf '%s\n' 'Nova tentativa em cinco segundos...'
    sleep 5 & espera=$!
    wait "$espera" || true
    espera=''
done
