import pytest

from core.ranges.range_parser import RangeParser

p = RangeParser()


def test_pair_exact():
    assert p.combo_count("AA") == 6


def test_suited_exact():
    assert p.combo_count("AKs") == 4


def test_offsuit_exact():
    assert p.combo_count("AKo") == 12


def test_both_suits():
    assert p.combo_count("AK") == 16


def test_pair_plus():
    # 77+ = 77 88 99 TT JJ QQ KK AA = 8 pairs × 6 = 48
    assert p.combo_count("77+") == 48


def test_pair_range():
    # 22-66 = 5 pairs × 6 = 30
    assert p.combo_count("22-66") == 30


def test_suited_plus():
    # AKs+ = only AKs (A is highest, no kicker above K before hitting pair)
    assert p.combo_count("AKs+") == 4
    # AJs+ = AJs, AQs, AKs = 12 (kicker J or better up to K)
    assert p.combo_count("AJs+") == 12


def test_connector_range_suited():
    # JTs-87s = JTs, T9s, 98s, 87s = 4 × 4 = 16
    assert p.combo_count("JTs-87s") == 16


def test_fixed_high_card_kicker_range_suited():
    # docs/AUDITORIA-2026-08-26.md item E6: A5s-A2s = A5s,A4s,A3s,A2s = 4x4=16
    assert p.combo_count("A5s-A2s") == 16
    assert sorted(p.parse("A5s-A2s")) == sorted(p.parse("A5s,A4s,A3s,A2s"))


def test_fixed_high_card_kicker_range_offsuit():
    # KQo-KTo = KQo,KJo,KTo = 3x12=36
    assert p.combo_count("KQo-KTo") == 36
    assert sorted(p.parse("KQo-KTo")) == sorted(p.parse("KQo,KJo,KTo"))


def test_fixed_high_card_kicker_range_reversed_order():
    # Same range spelled high-to-low on the left instead of low-to-high
    assert sorted(p.parse("A2s-A5s")) == sorted(p.parse("A5s-A2s"))


def test_empty_token():
    assert p.combo_count("") == 0


def test_multi_token():
    total = p.combo_count("AA,KK,QQ")
    assert total == 18


def test_no_duplicates():
    combos = p.parse("AA")
    seen = set()
    for c in combos:
        key = frozenset(c)
        assert key not in seen
        seen.add(key)


# ------------------------------------------------------------------
# Invalid tokens must raise, not silently disappear from the range
# ------------------------------------------------------------------


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        p.parse("AXs")


def test_garbage_token_raises():
    with pytest.raises(ValueError):
        p.parse("ABCDE")


def test_lone_plus_raises():
    with pytest.raises(ValueError):
        p.parse("+")


def test_pair_with_suit_suffix_raises():
    with pytest.raises(ValueError):
        p.parse("AAs")


def test_dash_range_invalid_rank_raises():
    with pytest.raises(ValueError):
        p.parse("JTs-8Xs")


def test_dash_range_mismatched_gap_raises():
    with pytest.raises(ValueError):
        p.parse("JTs-97s")


def test_dash_range_pair_as_connector_raises():
    with pytest.raises(ValueError):
        p.parse("JJs-88s")


def test_dash_range_missing_side_raises():
    with pytest.raises(ValueError):
        p.parse("JTs-")


def test_plus_combined_with_dash_raises():
    # docs/AUDITORIA-2026-08-26.md item E6: '+' used to be silently dropped
    # here (parsed as plain "22-66"), a silent-failure class the rest of the
    # parser deliberately avoids.
    with pytest.raises(ValueError):
        p.parse("22-66+")


def test_plus_combined_with_fixed_high_dash_raises():
    with pytest.raises(ValueError):
        p.parse("A5s-A2s+")
