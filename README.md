# Poker EV Calculator

Monorepo para cálculo de EV, análise GTO e visão computacional em poker.

## Estrutura

```
poker-ev-calculator/
├── core/          # Avaliação de mãos, equidade, ranges
├── engine/        # Cálculo de EV, ICM, GTO approximation
├── risk/          # Bankroll management, RoR, Kelly Criterion
├── simulation/    # Monte Carlo, cenários de torneio
├── vision/        # Detecção de mesa, OCR de cartas
├── output/        # Relatórios, gráficos, dashboards
└── cli/           # CLI (Typer + Rich): eval, equity, icm, roi
```

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
poker nash --stacks 3000,7000 --payouts 0.65,0.35
poker icm-pressure --stacks 5000,3000,2000 --payouts 0.5,0.3,0.2
poker roi --buyins 100,100,100 --cashes 0,200,500
poker bankroll --bankroll 5000 --winrate 5 --std 90
poker session --import history.txt --report
poker range-heatmap --range "22+,ATs+,KQo" --save heatmap.png
poker icm-chart --stacks 5000,3000,2000 --payouts 0.5,0.3,0.2 --save bubble.png
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
