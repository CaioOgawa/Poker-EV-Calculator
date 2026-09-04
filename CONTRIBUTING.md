# Contribuindo

Projeto pessoal, mas registrado aqui pra quem for mexer (inclusive eu, daqui a uns meses).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + charts + dev tooling
pip install -e ".[dev,vision]"   # também instala opencv/pytesseract/ultralytics (pesado, puxa torch)
pre-commit install               # ativa o hook de pre-commit (ruff + testes rápidos)
```

Requer Python 3.11+.

## Antes de commitar

```bash
make check      # lint + typecheck + test rápido
```

O pre-commit já roda `ruff` (lint + format) e a suíte rápida automaticamente a cada commit — `make check` é só pra rodar manualmente antes disso, ou pra incluir o `mypy` que o hook não cobre.

Comandos individuais em `make help`-equivalente (ver `Makefile`): `test`, `test-fast`, `test-cov`, `lint`, `fmt`, `typecheck`, `graph`.

## Convenções

Ver [`CONVENTIONS.md`](CONVENTIONS.md) — style, naming, notação de cartas, unidades de dinheiro/EV, padrão de testes, mensagens de commit.

## Estrutura

Ver a seção "Estrutura" do [`README.md`](README.md). Resumo: `core/` (avaliação de mãos/equity/ranges), `engine/` (EV/ICM/GTO), `risk/` (bankroll), `simulation/` (Monte Carlo), `vision/` (visão computacional), `output/` (relatórios/gráficos), `cli/` (comando `poker`), `data/` (parsers de hand history).

## CI

GitHub Actions roda lint + typecheck + testes rápidos em todo push/PR pra `main`, e a suíte completa (incluindo os testes `slow`) em push pra `main`.
