# Ambiente testado

Versões coletadas do ambiente virtual local em 19/11/2026 com `importlib.metadata`. Não constituem um arquivo de lock nem garantia para todas as versões permitidas por requirements.txt.

| Componente | Versão instalada |
|---|---|
| Python | 3.13.5 |
| Sistema operacional | Debian GNU/Linux |
| ultralytics | 8.4.190 |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| opencv-python | 5.0.0.93 |
| numpy | 2.5.4 |
| pillow | 13.0.0 |
| pyyaml | 6.0.3 |
| python-telegram-bot | 22.8 |
| python-dotenv | 1.2.4 |
| matplotlib | 3.11.2 |
| scipy | 1.18.1 |
| pi-heif | não instalado |

PyTorch informa build `2.14.1+cu130`, compilada com CUDA `13.0`. Os testes usam stubs e imagens sintéticas, sem inferência ou GPU. Para instalação em outra máquina, selecionar a build de torch adequada a CPU/GPU; nenhuma versão de torch ou CUDA foi fixada no requirements.

Os imports de terceiros encontrados em src/ e tests/ foram: cv2 (opencv-python), numpy, PIL (pillow), yaml (pyyaml), ultralytics, telegram (python-telegram-bot) e dotenv (python-dotenv). Todos já constam de requirements.txt. Torch é dependência transitiva do Ultralytics e é usado também nos roteiros históricos externos ao src/. Matplotlib e scipy não são imports diretos de src/ ou tests/: servem às análises externas. pi-heif não está instalado e HEIC não é suportado pelo bot.
