"""Expected Value calculator for push/fold and bet/call decisions."""
from __future__ import annotations


class EVCalculator:
    def push_fold_ev(
        self,
        equity: float,
        pot: float,
        effective_stack: float,
        fold_equity: float,
    ) -> float:
        """
        EV of shoving all-in.
        EV = fold_equity * pot + (1 - fold_equity) * (equity * (pot + effective_stack * 2) - effective_stack)
        """
        if not 0.0 <= equity <= 1.0:
            raise ValueError(f"equity must be in [0, 1], got {equity}")
        if not 0.0 <= fold_equity <= 1.0:
            raise ValueError(f"fold_equity must be in [0, 1], got {fold_equity}")
        ev_fold = fold_equity * pot
        ev_call = (1 - fold_equity) * (equity * (pot + effective_stack * 2) - effective_stack)
        return ev_fold + ev_call

    def call_ev(self, equity: float, pot: float, call_size: float) -> float:
        """EV of calling a bet. Positive = profitable call."""
        if not 0.0 <= equity <= 1.0:
            raise ValueError(f"equity must be in [0, 1], got {equity}")
        return equity * (pot + call_size) - call_size

    def pot_odds(self, call_size: float, pot: float) -> float:
        """Minimum equity needed to break even on a call."""
        return call_size / (pot + call_size)
