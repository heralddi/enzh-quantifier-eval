import collections
import json
import pathlib

import pytest

ITEMS = pathlib.Path(__file__).resolve().parents[1] / "data" / "items.jsonl"

REQUIRED_FIELDS = {
    "pair_id",
    "item_id",
    "language",
    "phenomenon",
    "context",
    "target",
    "cont_A",
    "cont_B",
    "forced",
    "note",
}
PHENOMENA = {"scalar_implicature", "scope", "numeral", "distributivity"}


@pytest.fixture(scope="module")
def rows():
    with open(ITEMS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_schema_and_counts(rows):
    assert len(rows) == 128
    assert len({r["item_id"] for r in rows}) == 128
    for r in rows:
        assert REQUIRED_FIELDS <= set(r), r["item_id"]
        assert r["language"] in {"en", "zh"}
        assert r["phenomenon"] in PHENOMENA
        assert r["forced"] in {"A", "B"}
        for field in ("context", "target", "cont_A", "cont_B"):
            assert isinstance(r[field], str) and r[field].strip()
        assert r["cont_A"] != r["cont_B"]


def test_pairing(rows):
    by_pair = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    assert len(by_pair) == 64
    for pid, members in by_pair.items():
        langs = sorted(m["language"] for m in members)
        assert langs == ["en", "zh"], pid
        assert len({m["phenomenon"] for m in members}) == 1, pid
        assert len({m["forced"] for m in members}) == 1, pid
    per_phen = collections.Counter(r["phenomenon"] for r in rows)
    assert all(v == 32 for v in per_phen.values())


def test_forced_balance(rows):
    counts = collections.Counter(
        (r["phenomenon"], r["language"], r["forced"]) for r in rows
    )
    for ph in PHENOMENA:
        for lang in ("en", "zh"):
            a, b = counts[(ph, lang, "A")], counts[(ph, lang, "B")]
            assert a + b == 16
            assert abs(a - b) <= 2, (ph, lang, a, b)


def test_target_wording_constraints(rows):
    for r in rows:
        ph, lang, tgt = r["phenomenon"], r["language"], r["target"]
        if ph == "distributivity" and lang == "en":
            low = tgt.lower()
            assert "together" not in low and "each" not in low, r["item_id"]
        if ph == "scalar_implicature":
            if lang == "en":
                assert "some" in tgt.lower(), r["item_id"]
            else:
                assert ("有些" in tgt) or ("一些" in tgt), r["item_id"]
        if ph == "numeral":
            assert ("three" in tgt.lower()) if lang == "en" else ("三" in tgt)
        if ph == "scope":
            assert ("every" in tgt.lower()) if lang == "en" else ("每" in tgt)
