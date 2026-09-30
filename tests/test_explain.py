import numpy as np

from sia.explain import occlusion


class KeywordModel:
    """Fake model: urgent iff the text contains 'outage'."""
    name = "fake"

    def predict_proba(self, descriptions, subjects=None, **kw):
        out = []
        for d in descriptions:
            out.append([0.05, 0.05, 0.85, 0.05] if "outage" in d.lower() else [0.8, 0.1, 0.05, 0.05])
        return np.array(out)


def test_occlusion_finds_the_decisive_word():
    text = "Hi Support, there is a full outage on our checkout page."
    ev = occlusion(KeywordModel(), text)
    assert ev and ev[0]["matched_text"] == "outage"
    assert text[ev[0]["start"]:ev[0]["end"]] == ev[0]["matched_text"]
    assert ev[0]["importance"] > 0.5


def test_occlusion_ignores_irrelevant_words():
    ev = occlusion(KeywordModel(), "Where is your headquarters located?")
    assert ev == []  # nothing single-handedly supports a confident Low verdict


def test_empty_text():
    assert occlusion(KeywordModel(), "") == []
