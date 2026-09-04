"""PokerStars cash-game hand history parser.

Scope: cash games only. Tournament headers (`Tournament #...`) use a
chip/ICM economy that doesn't map onto the dollar arithmetic below, so
they're rejected with a `ValueError` rather than silently misread — same
"fail loud" convention as `RangeParser`/`core/equity`. Other sites (GG,
PartyPoker) get their own module under `data/parsers/` when needed; nothing
here is site-agnostic on purpose.

Every real PokerStars hand history shows the account owner's hole cards via
a `Dealt to <name> [..]` line — that player is taken as hero for that hand,
so no name needs to be passed in.

Money handling: a player's net result for a hand is computed as
`collected - invested`, tracking every dollar that enters/leaves the pot
(blinds, calls, bets, raises — using each raise's own "to X" total rather
than its "size of raise" number, since those differ once a player already
has money in from an earlier action the same street — and uncalled-bet
returns). As a self-check, `sum(results) == -rake` must hold to the cent
for every hand (money in the pot went either to a stack or to the rake);
a violation means the parser misread the hand and raises immediately
instead of returning a plausible-looking wrong number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from output.reports.session_report import Hand, SessionReport

_AMT = r"[\$€£]?(?P<amt>[\d,]+\.?\d*)"

_HEADER_RE = re.compile(
    r"^PokerStars Hand #(?P<hand_id>\d+):.*?"
    r"\([\$€£]?(?P<sb>[\d,.]+)/[\$€£]?(?P<bb>[\d,.]+)(?:\s+\w+)?\)"
)
_TOURNAMENT_RE = re.compile(r"^PokerStars Hand #\d+: Tournament #")
_DEALT_RE = re.compile(r"^Dealt to (?P<player>.+?) \[(?P<cards>.+?)\]")
_STREET_RE = re.compile(r"^\*\*\* (FLOP|TURN|RIVER) \*\*\*")
_BOARD_CARDS_RE = re.compile(r"\[([^\]]+)\]")
_POST_RE = re.compile(
    r"^(?P<player>.+?): posts "
    r"(?:small blind|big blind|the ante|small & big blinds) " + _AMT
)
_ACTION_RE = re.compile(
    r"^(?P<player>.+?): (?P<verb>folds|checks|calls|bets|raises)"
    r"(?:\s+[\$€£]?(?P<amt1>[\d,]+\.?\d*))?"
    r"(?:\s+to\s+[\$€£]?(?P<amt2>[\d,]+\.?\d*))?"
)
_UNCALLED_RE = re.compile(r"^Uncalled bet \(" + _AMT + r"\) returned to (?P<player>.+)$")
_COLLECTED_RE = re.compile(
    r"^(?P<player>.+?) collected " + _AMT + r" from (?:the )?(?:main |side )?pot"
)
_SUMMARY_TOTAL_RE = re.compile(
    r"^Total pot [\$€£]?(?P<pot>[\d,]+\.?\d*).*?Rake [\$€£]?(?P<rake>[\d,]+\.?\d*)"
)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class ParsedHand:
    hand_id: str
    small_blind: float
    big_blind: float
    hero: str
    hero_cards: list[str]
    board: list[str]
    pot: float
    rake: float
    results: dict[str, float]  # player -> net $ won/lost this hand

    def to_hand(self, category: str | None = None) -> Hand:
        """Convert to the project's `Hand` (money in BB, per docs/ROADMAP.md §6).

        `ev` is not computed here: a raw hand history has no decision-tree
        context (villain ranges, the bet-sizing tree actually faced) for
        `engine/ev` to price each action against — it's set to 0.0, an
        explicit placeholder rather than a guess, the same spirit as
        `SolverBridge`'s `NotImplementedBridge`. Per-decision EV needs a
        separate pass that replays actions through `engine/ev`/`engine/gto`.
        """
        return Hand(
            hero_cards=self.hero_cards,
            board=self.board,
            pot=round(self.pot / self.big_blind, 4),
            result=round(self.results.get(self.hero, 0.0) / self.big_blind, 4),
            ev=0.0,
            category=category,
        )


class PokerStarsParser:
    """Parses PokerStars cash-game hand history text (`.txt` hand history exports)."""

    def parse_file(self, path: str | Path) -> list[ParsedHand]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text)

    def parse_text(self, text: str) -> list[ParsedHand]:
        blocks = re.split(r"\n(?=PokerStars (?:Hand|Game) #)", text.strip())
        return [self._parse_hand(b) for b in blocks if b.strip()]

    def _parse_hand(self, block: str) -> ParsedHand:
        lines = block.splitlines()
        header = lines[0].strip()
        if _TOURNAMENT_RE.match(header):
            raise ValueError(f"Tournament hand histories are not supported: {header!r}")
        m = _HEADER_RE.match(header)
        if not m:
            raise ValueError(f"Unrecognized PokerStars header: {header!r}")

        hand_id = m.group("hand_id")
        try:
            return self._parse_body(lines[1:], hand_id, _num(m.group("sb")), _num(m.group("bb")))
        except ValueError as e:
            raise ValueError(f"Hand #{hand_id}: {e}") from e

    def _parse_body(
        self, lines: list[str], hand_id: str, small_blind: float, big_blind: float
    ) -> ParsedHand:
        hero: str | None = None
        hero_cards: list[str] = []
        board: list[str] = []
        invested: dict[str, float] = {}
        street_contrib: dict[str, float] = {}
        collected: dict[str, float] = {}
        pot: float | None = None
        rake = 0.0
        in_summary = False

        def add_invested(player: str, amount: float) -> None:
            invested[player] = invested.get(player, 0.0) + amount

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("*** SUMMARY ***"):
                in_summary = True
                continue
            if in_summary:
                sm = _SUMMARY_TOTAL_RE.match(line)
                if sm:
                    pot = _num(sm.group("pot"))
                    rake = _num(sm.group("rake"))
                continue

            sm2 = _STREET_RE.match(line)
            if sm2:
                street_contrib.clear()
                board = " ".join(_BOARD_CARDS_RE.findall(line)).split()
                continue

            dm = _DEALT_RE.match(line)
            if dm:
                hero = dm.group("player")
                hero_cards = dm.group("cards").split()
                continue

            pm = _POST_RE.match(line)
            if pm:
                amt = _num(pm.group("amt"))
                add_invested(pm.group("player"), amt)
                street_contrib[pm.group("player")] = (
                    street_contrib.get(pm.group("player"), 0.0) + amt
                )
                continue

            um = _UNCALLED_RE.match(line)
            if um:
                add_invested(um.group("player"), -_num(um.group("amt")))
                continue

            cm = _COLLECTED_RE.match(line)
            if cm:
                player = cm.group("player")
                collected[player] = collected.get(player, 0.0) + _num(cm.group("amt"))
                continue

            am = _ACTION_RE.match(line)
            if am:
                player = am.group("player")
                verb = am.group("verb")
                if verb in ("folds", "checks"):
                    continue
                if verb == "raises":
                    if am.group("amt2") is None:
                        raise ValueError(f"Unrecognized raise line: {line!r}")
                    new_total = _num(am.group("amt2"))
                    delta = new_total - street_contrib.get(player, 0.0)
                    street_contrib[player] = new_total
                else:  # calls / bets
                    if am.group("amt1") is None:
                        raise ValueError(f"Unrecognized {verb} line: {line!r}")
                    delta = _num(am.group("amt1"))
                    street_contrib[player] = street_contrib.get(player, 0.0) + delta
                add_invested(player, delta)
                continue

        if hero is None:
            raise ValueError("no 'Dealt to' line found (hero hole cards missing)")
        if pot is None:
            raise ValueError("no 'Total pot' line found in *** SUMMARY ***")

        players = set(invested) | set(collected)
        results = {p: round(collected.get(p, 0.0) - invested.get(p, 0.0), 6) for p in players}

        invested_total = round(sum(invested.values()), 2)
        if abs(invested_total - pot) > 0.02:
            raise ValueError(
                f"money not conserved: tracked contributions sum to {invested_total}, "
                f"but 'Total pot' says {pot} — parser misread an action"
            )

        results_total = round(sum(results.values()), 2)
        if abs(results_total + rake) > 0.02:
            raise ValueError(
                f"money not conserved: sum(results)={results_total}, rake={rake} "
                "(expected sum(results) == -rake within rounding — parser misread an action)"
            )

        return ParsedHand(
            hand_id=hand_id,
            small_blind=small_blind,
            big_blind=big_blind,
            hero=hero,
            hero_cards=hero_cards,
            board=board,
            pot=pot,
            rake=rake,
            results=results,
        )


def hands_to_session(hands: list[ParsedHand], session_id: str | None = None) -> SessionReport:
    """Build a `SessionReport` from parsed hands, one `Hand` per parsed hand."""
    report = SessionReport() if session_id is None else SessionReport(session_id=session_id)
    for h in hands:
        report.add_hand(h.to_hand())
    return report
