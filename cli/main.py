"""Poker EV Calculator CLI — eval, equity, icm, roi commands."""
from __future__ import annotations

import json as _json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Poker EV Calculator")
console = Console()

_SUITS = frozenset("shdc")
_RANKS = frozenset("23456789TJQKA")


def _die(message: str) -> None:
    """Print a clean error and exit, instead of letting a traceback leak."""
    console.print(f"[red bold]Error:[/] {message}")
    raise typer.Exit(code=1)


def _parse_cards(s: str) -> list[str]:
    """Split a run of card tokens ('AsKs') into ['As', 'Ks']."""
    if len(s) % 2 != 0:
        raise typer.BadParameter(f"Invalid card string: {s!r}")
    cards = [s[i : i + 2] for i in range(0, len(s), 2)]
    for card in cards:
        if card[0] not in _RANKS or card[1] not in _SUITS:
            raise typer.BadParameter(f"Invalid card {card!r} in {s!r}")
    if len(set(cards)) != len(cards):
        raise typer.BadParameter(f"Duplicate card in {s!r}")
    return cards


def _is_hand(s: str) -> bool:
    """Return True if s looks like exactly two specific cards (e.g. 'AsKd')."""
    return (
        len(s) == 4
        and s[0] in _RANKS and s[1] in _SUITS
        and s[2] in _RANKS and s[3] in _SUITS
        and s[:2] != s[2:]
    )


@app.command("eval")
def eval_hand(
    hero: str = typer.Option(..., "--hero", help="Hole cards, e.g. AsKs"),
    board: str = typer.Option(..., "--board", help="Board cards (3-5), e.g. QhJsTs"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Evaluate hand strength on a given board."""
    from treys import Card, Evaluator

    from core.hand_evaluator import HandEvaluator

    hole = _parse_cards(hero)
    board_cards = _parse_cards(board)

    if len(hole) != 2:
        raise typer.BadParameter(f"--hero must be exactly 2 cards, got {len(hole)}")
    if len(board_cards) not in (3, 4, 5):
        raise typer.BadParameter(f"--board must be 3-5 cards, got {len(board_cards)}")

    evaluator = HandEvaluator()
    try:
        rank = evaluator.rank(hole, board_cards)
        pct = evaluator.percentile(hole, board_cards)
    except (ValueError, KeyError) as e:
        _die(str(e))

    te = Evaluator()
    class_name = te.class_to_string(te.get_rank_class(rank))

    result = {
        "hand": hero,
        "board": board,
        "rank": rank,
        "class": class_name,
        "percentile": round(pct * 100, 1),
    }

    if json_output:
        typer.echo(_json.dumps(result))
        return

    Card.print_pretty_cards([Card.new(c) for c in hole + board_cards])
    table = Table(title=f"Hand: {hero} | Board: {board}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Hand Class", class_name)
    table.add_row("Treys Rank", str(rank))
    table.add_row("Percentile", f"Top {result['percentile']}%")
    console.print(table)


@app.command()
def equity(
    hero: str = typer.Option(..., "--hero", help="Hero hole cards, e.g. AsKs"),
    villain: str = typer.Option(..., "--villain", help="Villain hand or range, e.g. QQ+,AKs"),
    board: Optional[str] = typer.Option(None, "--board", help="Board cards (optional)"),
    iterations: int = typer.Option(10_000, "--iterations", help="Monte Carlo iterations"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Calculate equity vs a specific hand or a range."""
    from core.equity import EquityCalculator

    hero_cards = _parse_cards(hero)
    board_cards = _parse_cards(board) if board else None

    if len(hero_cards) != 2:
        raise typer.BadParameter(f"--hero must be exactly 2 cards, got {len(hero_cards)}")
    if board_cards is not None and len(board_cards) not in (3, 4, 5):
        raise typer.BadParameter(f"--board must be 3-5 cards, got {len(board_cards)}")

    calc = EquityCalculator(iterations=iterations)

    try:
        if _is_hand(villain):
            villain_cards = [villain[:2], villain[2:]]
            eq = calc.heads_up(hero_cards, villain_cards, board=board_cards)
            mode = "vs hand"
        else:
            eq = calc.vs_range(hero_cards, villain, board=board_cards)
            mode = "vs range"
    except (ValueError, KeyError) as e:
        _die(str(e))

    result = {
        "hero": hero,
        "villain": villain,
        "mode": mode,
        "board": board or "",
        "equity": round(eq * 100, 1),
    }

    if json_output:
        typer.echo(_json.dumps(result))
        return

    console.print(
        f"[bold]{hero}[/] equity {mode} [bold]{villain}[/]"
        + (f" | board: {board}" if board else "")
        + f":  [green bold]{result['equity']}%[/]"
    )


@app.command()
def icm(
    stacks: str = typer.Option(..., "--stacks", help="Comma-separated stack sizes, e.g. 5000,3000,2000"),
    payouts: str = typer.Option(..., "--payouts", help="Comma-separated payout fractions, e.g. 0.5,0.3,0.2"),
    total: float = typer.Option(1000.0, "--total", help="Total prize pool in $"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Calculate ICM equity for each player at the table."""
    from engine.icm import ICMModel

    try:
        stack_list = [float(s) for s in stacks.split(",")]
        payout_list = [float(p) for p in payouts.split(",")]
    except ValueError:
        raise typer.BadParameter("--stacks and --payouts must be comma-separated numbers")

    model = ICMModel()
    try:
        equities = model.equity(stack_list, payout_list)
    except (ValueError, ZeroDivisionError) as e:
        _die(str(e))

    rows = [
        {
            "player": i + 1,
            "stack": s,
            "icm_fraction": round(e, 6),
            "icm_value": round(e * total, 2),
        }
        for i, (s, e) in enumerate(zip(stack_list, equities))
    ]

    if json_output:
        typer.echo(_json.dumps(rows))
        return

    table = Table(title=f"ICM — prize pool: ${total:,.0f}")
    table.add_column("Player", justify="center")
    table.add_column("Stack", justify="right")
    table.add_column("ICM %", justify="right")
    table.add_column("$ Value", justify="right", style="green")
    for r in rows:
        table.add_row(
            str(r["player"]),
            f"{r['stack']:,.0f}",
            f"{r['icm_fraction'] * 100:.2f}%",
            f"${r['icm_value']:,.2f}",
        )
    console.print(table)


@app.command()
def roi(
    buyins: str = typer.Option(..., "--buyins", help="Comma-separated buy-ins, e.g. 100,100,100"),
    cashes: str = typer.Option(..., "--cashes", help="Comma-separated cash amounts, e.g. 0,200,500"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Calculate tournament ROI and ITM rate."""
    from risk.roi.calculator import ROICalculator

    try:
        buyin_list = [float(b) for b in buyins.split(",")]
        cash_list = [float(c) for c in cashes.split(",")]
    except ValueError:
        raise typer.BadParameter("--buyins and --cashes must be comma-separated numbers")

    calc = ROICalculator()
    try:
        roi_val = calc.roi(buyin_list, cash_list)
        itm = calc.itm_rate(cash_list)
    except ValueError as e:
        _die(str(e))

    result = {
        "tournaments": len(buyin_list),
        "total_invested": sum(buyin_list),
        "total_cashed": sum(cash_list),
        "roi": round(roi_val * 100, 1),
        "itm_rate": round(itm * 100, 1),
        "profit": round(sum(cash_list) - sum(buyin_list), 2),
    }

    if json_output:
        typer.echo(_json.dumps(result))
        return

    roi_color = "green" if roi_val >= 0 else "red"
    console.print(f"Tournaments : {result['tournaments']}")
    console.print(f"Invested    : ${result['total_invested']:,.2f}")
    console.print(f"Cashed      : ${result['total_cashed']:,.2f}")
    console.print(f"Profit      : ${result['profit']:,.2f}")
    console.print(f"ROI         : [{roi_color}]{result['roi']}%[/]")
    console.print(f"ITM         : {result['itm_rate']}%")


if __name__ == "__main__":
    app()
