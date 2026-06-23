import math
from types import SimpleNamespace

import torch

from src.score import continuation_logprob, score_item

VOCAB = 50


class DummyTokenizer:
    """Character tokenizer over a 50-id vocabulary."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % VOCAB for c in text]}


class DummyModel:
    """Returns fixed logits. A favoured token id gets a higher logit."""

    device = "cpu"

    def __init__(self, favoured_id=None, bonus=5.0):
        self.favoured_id = favoured_id
        self.bonus = bonus

    def __call__(self, ids):
        batch, length = ids.shape
        logits = torch.zeros(batch, length, VOCAB)
        if self.favoured_id is not None:
            logits[:, :, self.favoured_id] = self.bonus
        return SimpleNamespace(logits=logits)


def test_uniform_logprob_matches_closed_form():
    model = DummyModel()
    tok = DummyTokenizer()
    cont = "abcd"
    lp, n = continuation_logprob(model, tok, "prefix text", cont)
    assert n == len(cont)
    assert math.isclose(lp, n * math.log(1.0 / VOCAB), rel_tol=1e-6)


def test_score_item_prefers_favoured_continuation():
    favoured = ord("x") % VOCAB
    model = DummyModel(favoured_id=favoured)
    tok = DummyTokenizer()
    row = {
        "item_id": "toy_en",
        "pair_id": "toy",
        "language": "en",
        "phenomenon": "numeral",
        "context": "Context.",
        "target": "Target.",
        "cont_A": "xxxx",
        "cont_B": "qqqq",
        "forced": "A",
    }
    rec = score_item(model, tok, row)
    assert rec["logprob_A"] > rec["logprob_B"]
    assert rec["choice"] == "A"
    assert rec["congruent"] == 1
    row_flipped = dict(row, cont_A="qqqq", cont_B="xxxx", forced="A")
    rec2 = score_item(model, tok, row_flipped)
    assert rec2["choice"] == "B"
    assert rec2["congruent"] == 0
