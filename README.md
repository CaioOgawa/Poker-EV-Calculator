# Poker EV Calculator

Monorepo para cálculo de EV, análise GTO e visão computacional em poker.

## Estrutura

```
poker-ev-calculator/
├── core/          # Avaliação de mãos, equidade, ranges (poker-expert)
├── engine/        # Cálculo de EV, ICM, GTO approximation (poker-quant)
├── risk/          # Bankroll management, RoR, Kelly Criterion (poker-quant)
├── simulation/    # Monte Carlo, cenários de torneio (poker-quant)
├── vision/        # Detecção de mesa, OCR de cartas (poker-vision)
├── output/        # Relatórios, gráficos, dashboards (poker-expert)
└── cli/           # CLI (Typer + Rich): eval, equity, icm, roi
```

## Skills

| Módulo | Skill |
|--------|-------|
| core, output | poker-expert |
| engine, risk, simulation | poker-quant |
| vision | poker-vision |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + charts + dev tooling
pip install -e ".[dev,vision]"   # também instala opencv/pytesseract/Pillow
```

Requer Python 3.11+.

## CLI

Instalado como o comando `poker` (via `pip install -e .`):

```bash
poker eval --hero AsKs --board QhJsTs
poker equity --hero AsKs --villain "QQ+,AKs" --board QhJsTs
poker icm --stacks 5000,3000,2000 --payouts 0.5,0.3,0.2 --total 1000
poker roi --buyins 100,100,100 --cashes 0,200,500
```

Todos os comandos aceitam `--json` para saída em JSON (bom para scripts/pipes).

## Uso via biblioteca

```python
from engine.ev import EVCalculator
from core.hand_evaluator import HandEvaluator
from core.equity import EquityCalculator
from engine.icm import ICMModel, ICMNash, ICMPressure
from risk.bankroll import KellyCriterion, BankrollSimulator, StakeRecommender
from simulation.montecarlo import MonteCarloSim
from simulation.scenarios.tournament import TournamentSim
```

## Desenvolvimento

```bash
make install      # cria .venv e instala tudo
make test         # suíte completa (inclui Monte Carlo / equity sweeps)
make test-fast    # só os testes rápidos (-m "not slow")
make lint         # ruff check
make fmt          # ruff format
make typecheck    # mypy em core/ e engine/
make check        # lint + typecheck + test
```

Hooks de pre-commit (ruff + testes rápidos) rodam automaticamente a cada commit:

```bash
pre-commit install   # uma vez, após clonar
```

CI (GitHub Actions) roda o job rápido em todo push/PR, e a suíte completa em push para `main`.
