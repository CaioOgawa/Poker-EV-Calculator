"""Preflop raise EV templates — RFI, 3-bet, and 4-bet.

All three spots share the same shape: risk `raise_size` chips; if everyone
folds you win the pot as it stood before your raise, if called the hand is
assumed to run to showdown with `equity_if_called`. They're kept as distinct
named methods (rather than one bare `raise_ev` call) because what `pot` must
already contain differs at each stage, and getting that wrong silently
misprices the spot — see each method's docstring.
"""

from __future__ import annotations


class PreflopRaiseEV:
    def raise_ev(
        self,
        fold_equity: float,
        pot: float,
        raise_size: float,
        equity_if_called: float,
    ) -> float:
        """EV of a preflop raise that is not for stacks (see `rfi_ev`,
        `three_bet_ev`, `four_bet_ev` for the named, documented entry points).
        """
        if not 0.0 <= fold_equity <= 1.0:
            raise ValueError(f"fold_equity must be in [0, 1], got {fold_equity}")
        if not 0.0 <= equity_if_called <= 1.0:
            raise ValueError(f"equity_if_called must be in [0, 1], got {equity_if_called}")
        ev_fold = fold_equity * pot
        pot_if_called = pot + raise_size * 2
        ev_called = (1 - fold_equity) * (equity_if_called * pot_if_called - raise_size)
        return ev_fold + ev_called

    def rfi_ev(
        self,
        fold_equity: float,
        pot: float,
        open_size: float,
        equity_if_called: float,
    ) -> float:
        """EV of an unopened raise-first-in (RFI).

        `pot` = dead money already in the middle before you act (blinds,
        antes) — it does not include your own `open_size`.
        """
        return self.raise_ev(fold_equity, pot, open_size, equity_if_called)

    def three_bet_ev(
        self,
        fold_equity: float,
        pot: float,
        three_bet_size: float,
        equity_if_called: float,
    ) -> float:
        """EV of a 3-bet over an open-raise.

        `pot` must include the opener's raise (and any blinds/antes), but not
        your own `three_bet_size`.
        """
        return self.raise_ev(fold_equity, pot, three_bet_size, equity_if_called)

    def four_bet_ev(
        self,
        fold_equity: float,
        pot: float,
        four_bet_size: float,
        equity_if_called: float,
    ) -> float:
        """EV of a 4-bet over a 3-bet.

        `pot` must include the 3-bettor's raise (and everything dead before
        it), but not your own `four_bet_size`.
        """
        return self.raise_ev(fold_equity, pot, four_bet_size, equity_if_called)
