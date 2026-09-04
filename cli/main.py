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
        and s[0] in _RANKS
        and s[1] in _SUITS
        and s[2] in _RANKS
        and s[3] in _SUITS
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
    # `percentile` is a strength score (1.0 = nuts), so "Top X%" is the
    # complement — a Royal Flush is "Top 0.1%", not "Top 99.9%".
    table.add_row("Percentile", f"Top {100 - result['percentile']:.1f}%")
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
    stacks: str = typer.Option(
        ..., "--stacks", help="Comma-separated stack sizes, e.g. 5000,3000,2000"
    ),
    payouts: str = typer.Option(
        ..., "--payouts", help="Comma-separated payout fractions, e.g. 0.5,0.3,0.2"
    ),
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
def nash(
    stacks: str = typer.Option(
        ..., "--stacks", help="hero,villain chip stacks, e.g. 3000,7000 (hero acts first)"
    ),
    payouts: str = typer.Option(
        ..., "--payouts", help="Comma-separated payout fractions, e.g. 0.65,0.35"
    ),
    sb: int = typer.Option(50, "--sb", help="Small blind in chips"),
    bb: int = typer.Option(100, "--bb", help="Big blind in chips"),
    max_iter: int = typer.Option(12, "--max-iter", help="Solver iterations"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Heads-up ICM Nash push/fold ranges (hero = SB pushes, villain = BB calls)."""
    from engine.icm.nash import HAND_RANK, ICMNash

    try:
        stack_list = [int(float(s)) for s in stacks.split(",")]
        payout_list = [float(p) for p in payouts.split(",")]
    except ValueError:
        raise typer.BadParameter("--stacks and --payouts must be comma-separated numbers")

    if len(stack_list) != 2:
        raise typer.BadParameter(f"--stacks must have exactly 2 values, got {len(stack_list)}")

    solver = ICMNash()
    try:
        nash_result = solver.solve_hu(stack_list, payout_list, sb=sb, bb=bb, max_iter=max_iter)
    except (ValueError, ZeroDivisionError, IndexError) as e:
        _die(str(e))

    total_hands = len(HAND_RANK)
    push_pct = (nash_result.push_threshold + 1) / total_hands * 100
    call_pct = (nash_result.call_threshold + 1) / total_hands * 100

    result = {
        "hero_stack": stack_list[0],
        "villain_stack": stack_list[1],
        "push_range": ",".join(nash_result.push_range),
        "push_pct": round(push_pct, 1),
        "call_range": ",".join(nash_result.call_range),
        "call_pct": round(call_pct, 1),
    }

    if json_output:
        typer.echo(_json.dumps(result))
        return

    console.print(
        f"[bold]Hero[/] (SB, {stack_list[0]:,} chips) vs "
        f"[bold]Villain[/] (BB, {stack_list[1]:,} chips)"
    )
    table = Table()
    table.add_column("Range", style="bold")
    table.add_column("% of hands", justify="right")
    table.add_column("Hands")
    table.add_row("Push (hero)", f"{result['push_pct']}%", result["push_range"] or "(none)")
    table.add_row("Call (villain)", f"{result['call_pct']}%", result["call_range"] or "(none)")
    console.print(table)


@app.command("icm-pressure")
def icm_pressure(
    stacks: str = typer.Option(
        ..., "--stacks", help="Comma-separated stack sizes, e.g. 5000,3000,2000"
    ),
    payouts: str = typer.Option(
        ..., "--payouts", help="Comma-separated payout fractions, e.g. 0.5,0.3,0.2"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Bubble factor matrix — how much riskier a chip confrontation is under ICM."""
    from engine.icm.pressure import ICMPressure

    try:
        stack_list = [float(s) for s in stacks.split(",")]
        payout_list = [float(p) for p in payouts.split(",")]
    except ValueError:
        raise typer.BadParameter("--stacks and --payouts must be comma-separated numbers")

    pressure = ICMPressure()
    try:
        matrix = pressure.all_bubble_factors(stack_list, payout_list)
        ranking = pressure.pressure_ranking(stack_list, payout_list)
    except (ValueError, ZeroDivisionError) as e:
        _die(str(e))

    def clean(v: float) -> float | str:
        return "inf" if v == float("inf") else round(v, 3)

    result = {
        "matrix": [[clean(v) for v in row] for row in matrix],
        "ranking": [{"player": i + 1, "avg_bubble_factor": clean(bf)} for i, bf in ranking],
    }

    if json_output:
        typer.echo(_json.dumps(result))
        return

    n = len(stack_list)
    table = Table(title="Bubble Factor (row risks chips vs column)")
    table.add_column("")
    for j in range(n):
        table.add_column(f"P{j + 1}", justify="right")
    for i in range(n):
        table.add_row(f"P{i + 1}", *[str(clean(matrix[i][j])) for j in range(n)])
    console.print(table)

    console.print("\n[bold]Most pressured (highest avg BF first):[/]")
    for i, bf in ranking:
        console.print(f"  P{i + 1}: {clean(bf)}")


@app.command()
def roi(
    buyins: str = typer.Option(..., "--buyins", help="Comma-separated buy-ins, e.g. 100,100,100"),
    cashes: str = typer.Option(
        ..., "--cashes", help="Comma-separated cash amounts, e.g. 0,200,500"
    ),
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


@app.command()
def bankroll(
    bankroll_dollars: float = typer.Option(..., "--bankroll", help="Bankroll in $"),
    winrate: float = typer.Option(..., "--winrate", help="Win rate in BB/100"),
    std: float = typer.Option(..., "--std", help="Standard deviation in BB/100"),
    max_ror: float = typer.Option(0.05, "--max-ror", help="Max acceptable risk of ruin"),
    buy_in_bbs: int = typer.Option(100, "--buy-in-bbs", help="Buy-in size in big blinds"),
    hands_per_hour: int = typer.Option(70, "--hands-per-hour", help="Hands per hour"),
    show_all: bool = typer.Option(
        False, "--all", help="Show every standard stake, not just the recommendation"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Recommend the highest stake with risk of ruin below --max-ror."""
    from risk.bankroll.recommender import StakeRecommender

    recommender = StakeRecommender()
    try:
        if show_all:
            rows = recommender.all_stakes_analysis(
                bankroll_dollars, winrate, std, buy_in_bbs, hands_per_hour
            )
        else:
            rows = [
                recommender.recommend(
                    bankroll_dollars, winrate, std, max_ror, buy_in_bbs, hands_per_hour
                )
            ]
    except ValueError as e:
        _die(str(e))

    result = [
        {
            "stake": r.stake_name,
            "bb_size": r.bb_size,
            "bankroll_in_bb": round(r.bankroll_in_bb, 1),
            "buy_ins": round(r.buy_ins, 1),
            "ror": round(r.ror * 100, 4),
            "ev_per_hour": round(r.ev_per_hour, 2),
        }
        for r in rows
    ]

    if json_output:
        typer.echo(_json.dumps(result))
        return

    table = Table(title="Stake recommendation" if not show_all else "Bankroll by stake")
    table.add_column("Stake", style="bold")
    table.add_column("BB size", justify="right")
    table.add_column("Buy-ins", justify="right")
    table.add_column("RoR", justify="right")
    table.add_column("EV/hr", justify="right", style="green")
    for r in result:
        ror_color = "red" if r["ror"] > max_ror * 100 else "green"
        table.add_row(
            r["stake"],
            f"${r['bb_size']:,.2f}",
            f"{r['buy_ins']:,.1f}",
            f"[{ror_color}]{r['ror']}%[/]",
            f"${r['ev_per_hour']:,.2f}",
        )
    console.print(table)


@app.command()
def session(
    import_file: str = typer.Option(
        ..., "--import", help="PokerStars hand history file (cash games only)"
    ),
    report: bool = typer.Option(False, "--report", help="Show per-category leak breakdown"),
    pdf: Optional[str] = typer.Option(None, "--pdf", help="Also export a PDF report to this path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Import a PokerStars hand history and print a session summary (BB units)."""
    from data.parsers.pokerstars import PokerStarsParser, hands_to_session

    try:
        parsed = PokerStarsParser().parse_file(import_file)
    except FileNotFoundError:
        _die(f"File not found: {import_file}")
    except ValueError as e:
        _die(str(e))

    if not parsed:
        _die("No hands found in file")

    session_report = hands_to_session(parsed)
    summary = session_report.summary()

    if pdf:
        from output.reports.pdf_export import PDFExporter

        try:
            PDFExporter().export_session(session_report, pdf)
        except ImportError as e:
            _die(str(e))

    if json_output:
        typer.echo(_json.dumps(summary))
        return

    console.print(f"Session    : {summary['session_id']}")
    console.print(f"Hands      : {summary['hands_played']}")
    console.print(f"Result     : {summary['total_result']} bb")
    console.print(
        f"EV         : {summary['total_ev']} bb  "
        "[dim](decision-level EV not computed from hand history)[/]"
    )
    console.print(f"Luck-adj.  : {summary['luck_adjusted']} bb")

    if report:
        from output.reports.leak_report import LeakReport

        leaks = LeakReport().by_category(session_report.hands)
        table = Table(title="Leaks by category")
        table.add_column("Category")
        table.add_column("Hands", justify="right")
        table.add_column("Total result (bb)", justify="right")
        for leak in leaks:
            table.add_row(leak.category, str(leak.hands), str(leak.total_result))
        console.print(table)

    if pdf:
        console.print(f"[green]PDF written to {pdf}[/]")


@app.command("range-heatmap")
def range_heatmap(
    range_str: str = typer.Option(..., "--range", help="Range string, e.g. '22+,ATs+,KQo'"),
    save: str = typer.Option(..., "--save", help="Output image path, e.g. heatmap.png"),
    title: str = typer.Option("Range", "--title", help="Chart title"),
    cmap: str = typer.Option("YlOrRd", "--cmap", help="Matplotlib colormap name"),
) -> None:
    """Render a 13x13 starting-hand grid for a range to an image file."""
    try:
        from output.charts.range_heatmap import RangeHeatmap
    except ImportError as e:
        _die(f"{e} (install with: pip install -e '.[charts]')")

    heatmap = RangeHeatmap()
    try:
        matrix = heatmap.matrix_from_range(range_str)
    except ValueError as e:
        _die(str(e))

    try:
        heatmap.plot(matrix, title=title, cmap=cmap, save_path=save)
    except ImportError as e:
        _die(f"{e} (install with: pip install -e '.[charts]')")

    console.print(f"[green]Saved to {save}[/]")


@app.command("icm-chart")
def icm_chart(
    stacks: str = typer.Option(
        ..., "--stacks", help="Comma-separated stack sizes, e.g. 5000,3000,2000"
    ),
    payouts: str = typer.Option(
        ..., "--payouts", help="Comma-separated payout fractions, e.g. 0.5,0.3,0.2"
    ),
    save: str = typer.Option(..., "--save", help="Output image path, e.g. bubble.png"),
    title: str = typer.Option("ICM Bubble Factor", "--title", help="Chart title"),
    cmap: str = typer.Option("RdYlGn_r", "--cmap", help="Matplotlib colormap name"),
) -> None:
    """Render the ICM bubble-factor matrix (all players vs all players) to an image file."""
    try:
        from output.charts.icm_pressure_chart import ICMPressureChart
    except ImportError as e:
        _die(f"{e} (install with: pip install -e '.[charts]')")

    try:
        stack_list = [float(s) for s in stacks.split(",")]
        payout_list = [float(p) for p in payouts.split(",")]
    except ValueError:
        raise typer.BadParameter("--stacks and --payouts must be comma-separated numbers")

    chart = ICMPressureChart()
    try:
        chart.plot(stack_list, payout_list, title=title, cmap=cmap, save_path=save)
    except (ValueError, ZeroDivisionError, ImportError) as e:
        _die(str(e))

    console.print(f"[green]Saved to {save}[/]")


@app.command()
def sim(
    winrate: float = typer.Option(..., "--winrate", help="Win rate in BB/100"),
    std: float = typer.Option(..., "--std", help="Standard deviation in BB/100"),
    bankroll_bb: float = typer.Option(..., "--bankroll", help="Starting bankroll in BB"),
    hands_per_session: int = typer.Option(100, "--hands-per-session"),
    num_sessions: int = typer.Option(500, "--num-sessions"),
    num_careers: int = typer.Option(10_000, "--num-careers"),
    ruin_threshold: float = typer.Option(
        0.0, "--ruin-threshold", help="Bankroll (BB) counted as ruin"
    ),
    multitable: int = typer.Option(1, "--multitable", help="Tables played simultaneously"),
    seed: Optional[int] = typer.Option(42, "--seed", help="RNG seed (omit for non-deterministic)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON (includes full curves)"),
) -> None:
    """Monte Carlo bankroll career simulation (percentile curves + ruin rate)."""
    from risk.bankroll.simulator import BankrollSimulator

    simulator = BankrollSimulator()
    try:
        if multitable > 1:
            res = simulator.simulate_multitable(
                multitable,
                winrate,
                std,
                bankroll_bb,
                hands_per_session,
                num_sessions,
                num_careers,
                ruin_threshold,
                seed,
            )
        else:
            res = simulator.simulate(
                winrate,
                std,
                bankroll_bb,
                hands_per_session,
                num_sessions,
                num_careers,
                ruin_threshold,
                seed,
            )
    except ValueError as e:
        _die(str(e))

    total_hands = int(res.hands[-1])
    result = {
        "starting_bankroll_bb": bankroll_bb,
        "total_hands": total_hands,
        "num_careers": num_careers,
        "ruin_rate": round(res.ruin_rate * 100, 2),
        "mean_final_bb": round(res.mean_final, 1),
        "final_p5_bb": round(float(res.p5[-1]), 1),
        "final_p50_bb": round(float(res.p50[-1]), 1),
        "final_p95_bb": round(float(res.p95[-1]), 1),
    }
    if json_output:
        result["hands"] = res.hands.tolist()
        result["p5"] = res.p5.tolist()
        result["p50"] = res.p50.tolist()
        result["p95"] = res.p95.tolist()
        typer.echo(_json.dumps(result))
        return

    ruin_color = "red" if result["ruin_rate"] > 5.0 else "green"
    console.print(f"Starting bankroll : {bankroll_bb:,.0f} bb")
    console.print(
        f"Sessions          : {num_sessions:,} x {hands_per_session} hands "
        f"= {total_hands:,} hands" + (f" ({multitable} tables)" if multitable > 1 else "")
    )
    console.print(f"Careers simulated : {num_careers:,}")
    console.print(f"Ruin rate         : [{ruin_color}]{result['ruin_rate']}%[/]")
    console.print(f"Mean final        : {result['mean_final_bb']:,.1f} bb")
    console.print(
        "Final p5/p50/p95  : "
        f"{result['final_p5_bb']:,.1f} / {result['final_p50_bb']:,.1f} / "
        f"{result['final_p95_bb']:,.1f} bb"
    )


if __name__ == "__main__":
    app()
