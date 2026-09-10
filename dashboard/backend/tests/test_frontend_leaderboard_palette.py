"""Colour-encoding contracts for the Competition/Live board chart.

Twelve curves share one plot area: seven LLM models, three rule-based strategy
baselines, two market indices. The 2026-09-03 user feedback ("the labeling is
quite messy ... maybe try 'baseline strategy' 'Model' 'Market Index' with
different colors") is a legibility report on that plot, and the fix is a
division of labour between the two channels the chart has:

* **Hue carries model identity.** The seven models are what a reader actually
  compares, so they get the whole validated categorical palette.
* **Dash carries reference identity.** The five reference curves are neutral
  grey and differ from each other only by dash pattern.

That split is not a style preference, it is what measurement allows. Running the
dataviz palette validator against the dark chart surface (#131a35):

* the shipped palette FAILED -- worst adjacent normal-vision Delta E 11.2
  (#FB923C vs #FBBF24), against a >= 15 floor, plus every entry above the dark
  lightness band. Two model curves genuinely were hard to tell apart.
* a warm-hue-family palette for the models, which is what the feedback literally
  suggests, FAILED harder (normal-vision Delta E 7.1, CVD 4.8). Seven series do
  not fit in one hue family; that idea is arithmetically closed, not merely
  unfashionable.
* every attempt at a *desaturated but still colour-coded* baseline trio FAILED
  at normal-vision Delta E 0.9-9.1. Desaturated colours cannot carry identity at
  all -- which is why the baselines here carry none, and dash carries it instead.
* the validated eight-hue categorical order PASSES all five checks on both dark
  surfaces (#131a35 and #0a0e27).

These guards therefore pin the *invariants* that survived that measurement, not
the hexes for their own sake. Re-run the validator before changing any value
here: `node scripts/validate_palette.js "<hexes>" --mode dark --surface "#131a35"`
in the dataviz skill.
"""

import re
from pathlib import Path

_LEADERBOARD_JS = (
    Path(__file__).resolve().parents[2] / "frontend" / "js" / "leaderboard.js"
).read_text(encoding="utf-8")


def _const_block(name: str) -> str:
    """`const <name> = {...};` or `= [...];`, bracket-matched.

    A line-based read returns the first line of a multi-line literal, which is
    syntactically incomplete -- every assertion built on it is then vacuous.
    """
    start = _LEADERBOARD_JS.index(f"const {name}")
    opener = min(
        i
        for i in (
            _LEADERBOARD_JS.find("{", start),
            _LEADERBOARD_JS.find("[", start),
        )
        if i != -1
    )
    closer = {"{": "}", "[": "]"}[_LEADERBOARD_JS[opener]]
    depth, index = 0, opener
    while True:
        if _LEADERBOARD_JS[index] == _LEADERBOARD_JS[opener]:
            depth += 1
        elif _LEADERBOARD_JS[index] == closer:
            depth -= 1
            if depth == 0:
                return _LEADERBOARD_JS[start : index + 1]
        index += 1


# Max RGB max-min spread a curve may have and still count as neutral.
#
# Derived from the two populations rather than picked: the reference greys ship
# at a spread of 22-36 (a slate tint reads as grey, and #94A3B8 has shipped as
# the DJIA line for as long as the chart has existed), while the *least* saturated
# entry in MODEL_COLOR_PALETTE is #9085e9 at 100. Anything between 36 and 100 is
# a colour trying to be subtle, which is the state the measurements rule out. 45
# sits in that gap with margin on the side that matters -- it admits a grey with a
# slightly stronger tint, and rejects every hue in the model palette by 2x.
_NEUTRAL_SPREAD_MAX = 45


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _reference_entries() -> dict[str, dict]:
    """Every LEADERBOARD_STYLES entry, as {label: {"color", "dash"}}."""
    block = _const_block("LEADERBOARD_STYLES")
    entries = {}
    for match in re.finditer(
        r"(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z_$][\w$]*))\s*:\s*\{([^}]*)\}", block
    ):
        label = match.group(1) or match.group(2) or match.group(3)
        body = match.group(4)
        colour = re.search(r"color:\s*'(#[0-9A-Fa-f]{6})'", body)
        dash = re.search(r"dash:\s*(\[[^\]]*\])", body)
        entries[label] = {
            "color": colour.group(1) if colour else None,
            "dash": dash.group(1) if dash else None,
        }
    return entries


def test_the_model_palette_is_the_validated_eight_and_nothing_more():
    """Eight, not ten. The previous palette carried ten slots so that a board of
    seven models never wrapped, but four of those ten sat outside the dark
    lightness band and one adjacent pair was below the normal-vision floor --
    headroom bought with curves the reader cannot separate.

    The validated set is eight, and adding a ninth or tenth hue breaks it
    (measured: chroma floor and CVD both FAIL at ten). So the wrap threshold is
    now 8: a board that grows past eight models needs a re-validated palette,
    not two more colours appended here on the assumption that more is safer.
    """
    colours = re.findall(r"#[0-9A-Fa-f]{6}", _const_block("MODEL_COLOR_PALETTE"))
    assert len(colours) == 8, (
        "the model palette is the validated eight-hue categorical order; "
        "appending unvalidated hues silently reintroduces indistinguishable curves"
    )
    assert len({c.lower() for c in colours}) == 8, "duplicate colours"


def test_every_reference_curve_is_achromatic():
    """The baselines and indices carry no hue, because hue is the models' channel.

    This is the computable half of the feedback fix: a coloured baseline reads as
    a thirteenth competitor, and the three bright ones that shipped (#38BDF8,
    #C084FC, #4ADE80) collided with the model palette's blue, violet and green.
    Desaturating them instead of neutralising them does not work either -- every
    muted trio measured below the normal-vision floor, so a "muted but still
    colour-coded" baseline is a colour that claims to say something and doesn't.
    """
    for label, style in _reference_entries().items():
        red, green, blue = _rgb(style["color"])
        assert max(red, green, blue) - min(red, green, blue) <= _NEUTRAL_SPREAD_MAX, (
            f"{label} carries hue ({style['color']}); reference curves are grey "
            "so that hue means 'model' and nothing else"
        )


def test_every_reference_curve_has_its_own_dash_pattern():
    """Dash is the only channel left telling the five reference curves apart, so
    no two of them may share one.

    The three strategy baselines all shipped `[10, 6]`. That was survivable only
    while their colours differed; once colour stops carrying identity -- which
    the measurements above force -- a shared dash makes them the same curve to
    the reader, and "Buy & Hold vs Mean-Variance" becomes unanswerable from the
    chart. This is the assertion that makes the achromatic guard above safe.
    """
    entries = _reference_entries()
    dashes = [style["dash"] for style in entries.values()]
    assert all(dashes), f"a reference curve has no dash pattern: {entries}"
    assert len(set(dashes)) == len(dashes), (
        f"reference curves share a dash pattern and are now indistinguishable: {dashes}"
    )


def test_models_are_the_only_solid_curves():
    """The category cue a reader gets before reading any legend.

    `KIND_WIDTH`/`KIND_ALPHA` already make references thinner and fainter; what
    makes the classes *nameable* is that every reference curve is broken and
    every model curve is not.
    """
    for label, style in _reference_entries().items():
        assert style["dash"] not in ("[]", "[0]"), (
            f"{label} draws solid, which is the models' cue"
        )


def test_the_legend_names_its_groups_in_the_ranking_tables_vocabulary():
    """The legend already *sorted* by category and never *said* so.

    Sorted-but-unlabelled is why the feedback had to guess at the taxonomy
    ("maybe try 'baseline strategy' 'Model' 'Market Index' with different
    colors") -- the ranking table shows those three badges, the chart's key
    showed a flat list, and nothing connected them. Now that hue means "model"
    and grey means "reference", the key is the only place that fact is written
    down, so it has to be written in the *same words* the table uses.

    `formatEntryBadge` is the table's vocabulary. Asserting the legend's group
    labels are drawn from it is what stops the two from drifting into two names
    for one thing, which is the failure mode a hardcoded string here invites.
    """
    badge_body = _LEADERBOARD_JS[
        _LEADERBOARD_JS.index("function formatEntryBadge") : _LEADERBOARD_JS.index(
            "function isModelEntry"
        )
    ]
    table_vocabulary = set(re.findall(r"return '([^']+)'", badge_body))
    assert "Baseline Strategy" in table_vocabulary, "the table's vocabulary moved"

    group_labels = set(
        re.findall(r"'([^']+)'", _const_block("KIND_GROUP_LABEL").split("=", 1)[1])
    )
    # Asserted in this direction, not the reverse: `formatEntryBadge` passes an
    # unrecognised badge through unchanged, so the table's vocabulary is open and
    # cannot be enumerated to check the legend against. What *is* closed is the
    # set of names the badge function normalises TO -- those are the two it owns,
    # and both must appear in the legend's map or the surfaces have drifted.
    missing = table_vocabulary - group_labels
    assert not missing, (
        f"the ranking table normalises badges the legend never names: {sorted(missing)}"
    )

    legend = _LEADERBOARD_JS[
        _LEADERBOARD_JS.index("function buildCustomLegend") :
    ]
    legend = legend[: legend.index("\nasync function")]
    assert "KIND_GROUP_LABEL" in legend, (
        "the legend hardcodes its headings instead of reading the shared map"
    )
