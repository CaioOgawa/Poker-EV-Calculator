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
└── output/        # Relatórios, gráficos, dashboards (poker-expert)
```

## Skills

| Módulo | Skill |
|--------|-------|
| core, output | poker-expert |
| engine, risk, simulation | poker-quant |
| vision | poker-vision |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```python
from engine.ev import EVCalculator
from core.hand_evaluator import HandEvaluator
from simulation.montecarlo import MonteCarloSim
```
