"""Per-layout screen calibration: where hero's hole cards render on screen.

Why a fixed region instead of matching a seat marker's bbox: PokerStars —
like every poker client — always draws the local (logged-in) player's own
seat at a fixed position on screen (bottom-center) and rotates everyone
else's seats around it, so "where is hero" is a property of the *layout*,
not something to infer per hand.

Verified against `vision/data/raw/pokerstars/*/labels/`: the seat markers
`p0s`..`p5s` each cluster at stdev <= 0.003 (normalized coords) across all
447 training images — screen-position-anchored, not tied to a rotating
poker seat. The bottom slot (`p0s`, y ~= 0.947, closest to the viewer) is
the local-player position by PokerStars' own UI convention. Of the 447
images, 7 happen to show two cards in that exact slot; those cluster at
x=[0.449, 0.514], y=[0.818, 0.825], stdev(y)=0.0018 — essentially pixel-
fixed, same conclusion from the card side.

Honesty note: this training set is scraped third-party footage (some images
are an all-in-equity replayer overlay showing every seat's hole cards, not
a single logged-in player's own view — see the analysis in this module's
git history for a concrete example), so *which identity* sits in the
bottom slot varies image to image here. That doesn't undermine the
calibration: the geometric claim is only that PokerStars renders the
*local viewer's own seat* at this fixed position, which holds regardless
of whose footage it is. For a live capture against your own logged-in
client, that seat is always you. This hasn't been verified against a real
live table yet (see docs/ROADMAP.md) — treat it as the best defensible
default, not a closed loop.

A different room/skin/resolution needs its own calibration, hence the YAML
override path rather than only a hardcoded constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableCalibration:
    name: str
    hero_region: tuple[float, float, float, float]  # normalized (x1, y1, x2, y2)
    hero_seat: int | None = None  # index into GameState.seats, if known


# Measured from vision/data/raw/pokerstars/*/labels — see module docstring.
# Padded beyond the observed [0.449, 0.514] x [0.818, 0.825] card-center
# range to tolerate bbox extent (not just center) and minor aspect jitter.
POKERSTARS_6MAX = TableCalibration(
    name="pokerstars_6max",
    hero_region=(0.40, 0.78, 0.56, 0.87),
    hero_seat=0,
)

_BUILTIN = {"pokerstars_6max": POKERSTARS_6MAX}


def load_calibration(name_or_path: str | Path) -> TableCalibration:
    """Look up a built-in calibration by name, or load one from a YAML file
    with `name`, `hero_region: [x1, y1, x2, y2]`, and optional `hero_seat`."""
    key = str(name_or_path)
    if key in _BUILTIN:
        return _BUILTIN[key]

    import yaml

    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No built-in calibration {key!r} and no such file — built-ins: {sorted(_BUILTIN)}"
        )
    data = yaml.safe_load(path.read_text())
    region = data["hero_region"]
    if len(region) != 4:
        raise ValueError(f"hero_region must have 4 values (x1,y1,x2,y2), got {region}")
    return TableCalibration(
        name=data.get("name", path.stem),
        hero_region=(region[0], region[1], region[2], region[3]),
        hero_seat=data.get("hero_seat"),
    )


def attribute_hero_hand(
    exposed_cards: list[dict],
    img_width: float,
    img_height: float,
    calibration: TableCalibration = POKERSTARS_6MAX,
) -> list[str] | None:
    """Return the exposed cards whose centers fall in the calibrated hero
    region, or None unless that finds exactly 2 (no confident hero hand
    right now — folded, muck, or no hand dealt)."""
    x1, y1, x2, y2 = calibration.hero_region
    region = (x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height)

    in_region = []
    for c in exposed_cards:
        bx1, by1, bx2, by2 = c["bbox"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
            in_region.append(c)

    if len(in_region) != 2:
        return None
    return [c["card"] for c in in_region]
