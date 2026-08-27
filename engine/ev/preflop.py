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
        *,
        hero_invested: float = 0.0,
        villain_invested: float = 0.0,
    ) -> float:
        """EV of a preflop raise that is not for stacks (see `rfi_ev`,
        `three_bet_ev`, `four_bet_ev` for the named, documented entry points).

        `pot` is the money in the middle before this raise, by contract
        including anything hero or villain already committed in earlier
        streets of the betting (hero's own blind, villain's prior open, ...).
        `hero_invested` / `villain_invested` are the slices of that `pot`
        that belong to hero and to the player who will call, respectively —
        needed because raising only costs/wins the *incremental* chips, not
        the full `raise_size`/`pot` on top of money already there. Both
        default to 0 (open from a position with nothing in yet, callable by
        an opponent with nothing in yet).
        """
        if not 0.0 <= fold_equity <= 1.0:
            raise ValueError(f"fold_equity must be in [0, 1], got {fold_equity}")
        if not 0.0 <= equity_if_called <= 1.0:
            raise ValueError(f"equity_if_called must be in [0, 1], got {equity_if_called}")
        hero_cost = raise_size - hero_invested
        villain_cost = raise_size - villain_invested
        ev_fold = fold_equity * (pot - hero_invested)
        pot_if_called = pot + hero_cost + villain_cost
        ev_called = (1 - fold_equity) * (equity_if_called * pot_if_called - hero_cost)
        return ev_fold + ev_called

    def rfi_ev(
        self,
        fold_equity: float,
        pot: float,
        open_size: float,
        equity_if_called: float,
        *,
        villain_invested: float = 0.0,
    ) -> float:
        """EV of an unopened raise-first-in (RFI).

        `pot` = dead money already in the middle before you act (blinds,
        antes) — it does not include your own `open_size` (hero has nothing
        invested yet). `villain_invested` is the blind already posted by
        whoever calls: 1.0 if the BB calls, 0.5 if the SB calls, 0.0 if a
        blind-less player (already folded back to you, or straddle) calls.
        """
        return self.raise_ev(
            fold_equity, pot, open_size, equity_if_called, villain_invested=villain_invested
        )

    def three_bet_ev(
        self,
        fold_equity: float,
        pot: float,
        three_bet_size: float,
        equity_if_called: float,
        *,
        hero_invested: float = 0.0,
        villain_invested: float = 0.0,
    ) -> float:
        """EV of a 3-bet over an open-raise.

        `pot` must include the opener's raise (and any blinds/antes). Pass
        `villain_invested` = that opener's `open_size` (the money of theirs
        already in `pot` that they don't have to add again to call). Pass
        `hero_invested` = hero's own blind already in `pot` if 3-betting from
        the SB/BB (0 from CO/BTN/etc., where hero has nothing in yet).
        """
        return self.raise_ev(
            fold_equity,
            pot,
            three_bet_size,
            equity_if_called,
            hero_invested=hero_invested,
            villain_invested=villain_invested,
        )

    def four_bet_ev(
        self,
        fold_equity: float,
        pot: float,
        four_bet_size: float,
        equity_if_called: float,
        *,
        hero_invested: float = 0.0,
        villain_invested: float = 0.0,
    ) -> float:
        """EV of a 4-bet over a 3-bet.

        `pot` must include the 3-bettor's raise (and everything dead before
        it). Pass `hero_invested` = hero's own open_size already in `pot`
        (hero was the original raiser), `villain_invested` = the 3-bettor's
        `three_bet_size` already in `pot`.
        """
        return self.raise_ev(
            fold_equity,
            pot,
            four_bet_size,
            equity_if_called,
            hero_invested=hero_invested,
            villain_invested=villain_invested,
        )
