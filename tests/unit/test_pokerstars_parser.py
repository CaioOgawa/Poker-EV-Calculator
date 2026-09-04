from __future__ import annotations

import pytest

from data.parsers.pokerstars import PokerStarsParser, hands_to_session

# Hero (BTN) 3-bets... no, single raise preflop, bets flop, bets turn and
# takes it down when villain folds. Numbers below are hand-verified: every
# dollar posted is accounted for, sum(results) == -rake == 0.
MULTISTREET_HAND = """\
PokerStars Hand #123456789: Hold'em No Limit ($0.05/$0.10 USD) - 2024/01/01 12:00:00 ET
Table 'Test I' 6-max Seat #3 is the button
Seat 1: Player1 ($10.00 in chips)
Seat 2: Player2 ($10.00 in chips)
Seat 3: Hero ($10.00 in chips)
Player1: posts small blind $0.05
Player2: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [As Ks]
Hero: raises $0.20 to $0.30
Player1: folds
Player2: calls $0.20
*** FLOP *** [Ah Kd 2c]
Player2: checks
Hero: bets $0.40
Player2: calls $0.40
*** TURN *** [Ah Kd 2c] [7s]
Player2: checks
Hero: bets $0.90
Player2: folds
Uncalled bet ($0.90) returned to Hero
Hero collected $1.45 from pot
*** SUMMARY ***
Total pot $1.45 | Rake $0.00
Board [Ah Kd 2c 7s]
Seat 1: Player1 (small blind) folded before Flop
Seat 2: Player2 (big blind) folded on the Turn
Seat 3: Hero (button) collected ($1.45)
"""

# Hero (SB) folds preflop uncontested to the BB's raise-free walk-off — no
# board is ever dealt.
WALK_HAND = """\
PokerStars Hand #200000001: Hold'em No Limit ($0.05/$0.10 USD) - 2024/01/01 12:05:00 ET
Table 'Test I' 6-max Seat #1 is the button
Seat 1: Hero ($10.00 in chips)
Seat 2: Player2 ($10.00 in chips)
Hero: posts small blind $0.05
Player2: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [7c 2d]
Hero: folds
Player2 collected $0.15 from pot
*** SUMMARY ***
Total pot $0.15 | Rake $0.00
Seat 1: Hero (button) (small blind) folded before Flop
Seat 2: Player2 (big blind) collected ($0.15)
"""

# Hero shoves all-in preflop from a short stack; villain folds and the
# excess over villain's BB is returned uncalled.
ALLIN_HAND = """\
PokerStars Hand #400000001: Hold'em No Limit ($0.05/$0.10 USD) - 2024/01/02 09:00:00 ET
Table 'Test I' 6-max Seat #4 is the button
Seat 4: Hero ($5.00 in chips)
Seat 5: Player2 ($10.00 in chips)
Hero: posts small blind $0.05
Player2: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [Ac Ad]
Hero: raises $4.90 to $5 and is all-in
Player2: folds
Uncalled bet ($4.90) returned to Hero
Hero collected $0.20 from pot
*** SUMMARY ***
Total pot $0.20 | Rake $0.00
Seat 4: Hero (button) (small blind) collected ($0.20)
Seat 5: Player2 (big blind) folded before Flop
"""

TOURNAMENT_HAND = (
    "PokerStars Hand #300000001: Tournament #999999999, $10+$1 USD Hold'em No Limit "
    "- Level I (10/20) - 2024/01/01 12:00:00 ET\n"
)


def test_parses_multistreet_hand():
    [hand] = PokerStarsParser().parse_text(MULTISTREET_HAND)

    assert hand.hand_id == "123456789"
    assert hand.small_blind == 0.05
    assert hand.big_blind == 0.10
    assert hand.hero == "Hero"
    assert hand.hero_cards == ["As", "Ks"]
    assert hand.board == ["Ah", "Kd", "2c", "7s"]
    assert hand.pot == 1.45
    assert hand.rake == 0.0
    assert hand.results["Hero"] == pytest.approx(0.75)
    assert hand.results["Player1"] == pytest.approx(-0.05)
    assert hand.results["Player2"] == pytest.approx(-0.70)


def test_parses_walk_hand_with_empty_board():
    [hand] = PokerStarsParser().parse_text(WALK_HAND)

    assert hand.board == []
    assert hand.results["Hero"] == pytest.approx(-0.05)
    assert hand.results["Player2"] == pytest.approx(0.05)


def test_raise_uses_to_amount_not_raise_size():
    [hand] = PokerStarsParser().parse_text(ALLIN_HAND)

    assert hand.results["Hero"] == pytest.approx(0.10)
    assert hand.results["Player2"] == pytest.approx(-0.10)


def test_rejects_tournament_hands():
    with pytest.raises(ValueError, match="Tournament"):
        PokerStarsParser().parse_text(TOURNAMENT_HAND)


def test_rejects_unrecognized_header():
    with pytest.raises(ValueError, match="Unrecognized PokerStars header"):
        PokerStarsParser().parse_text("PokerStars Hand #1: something without stakes parens\n")


def test_money_not_conserved_raises():
    tampered = MULTISTREET_HAND.replace("Total pot $1.45", "Total pot $99.00")
    with pytest.raises(ValueError, match="money not conserved"):
        PokerStarsParser().parse_text(tampered)


def test_parse_file_reads_multiple_hands(tmp_path):
    path = tmp_path / "history.txt"
    path.write_text(MULTISTREET_HAND + "\n" + WALK_HAND)

    hands = PokerStarsParser().parse_file(path)

    assert len(hands) == 2
    assert [h.hand_id for h in hands] == ["123456789", "200000001"]


def test_to_hand_converts_dollars_to_big_blinds():
    [hand] = PokerStarsParser().parse_text(MULTISTREET_HAND)
    h = hand.to_hand()

    assert h.hero_cards == ["As", "Ks"]
    assert h.board == ["Ah", "Kd", "2c", "7s"]
    assert h.pot == pytest.approx(14.5)
    assert h.result == pytest.approx(7.5)
    assert h.ev == 0.0
    assert h.category is None


def test_hands_to_session_builds_report_in_big_blinds():
    hands = PokerStarsParser().parse_text(MULTISTREET_HAND + "\n" + WALK_HAND)
    report = hands_to_session(hands, session_id="test-session")

    summary = report.summary()
    assert summary["session_id"] == "test-session"
    assert summary["hands_played"] == 2
    # 7.5 bb (multistreet win) + (-0.5 bb) (walk fold) = 7.0 bb
    assert summary["total_result"] == pytest.approx(7.0)
