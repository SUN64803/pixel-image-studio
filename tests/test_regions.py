"""Tests for regional prompt normalisation."""
import sana_engine as se


def test_normalize_regions_valid_entry():
    out = se.normalize_regions(
        [
            {
                "prompt": "  a red   fox  ",
                "weight": 1.2,
                "x": 0.0,
                "y": 0.0,
                "w": 0.5,
                "h": 1.0,
                "label": " left ",
            }
        ]
    )
    assert len(out) == 1
    r = out[0]
    assert r["prompt"] == "a red fox"
    assert r["weight"] == 1.2
    assert r["x"] == 0.0 and r["y"] == 0.0
    assert r["w"] == 0.5 and r["h"] == 1.0
    assert r["label"] == "left"


def test_normalize_regions_clamps():
    out = se.normalize_regions(
        [{"prompt": "x", "weight": 99, "x": -1, "y": 0.9, "w": 5, "h": 5}]
    )
    r = out[0]
    assert r["weight"] == 1.5
    assert r["x"] == 0.0
    assert r["y"] == 0.9
    # w must fit inside [x, 1] and h inside [y, 1]
    assert r["w"] <= 1.0 - r["x"] + 1e-9
    assert r["h"] <= 1.0 - r["y"] + 1e-9


def test_normalize_regions_min_size_floor():
    # y at the very bottom edge → h collapses to the 0.02 floor (engine's rule)
    out = se.normalize_regions([{"prompt": "edge", "y": 1.0, "x": 0, "w": 1, "h": 1}])
    r = out[0]
    assert r["y"] == 1.0
    assert r["h"] == 0.02


def test_normalize_regions_skips_junk():
    out = se.normalize_regions(
        [
            {"prompt": "   ", "x": 0, "y": 0, "w": 1, "h": 1},  # empty prompt
            {"prompt": "ok", "x": "nan-ish", "y": 0, "w": 1, "h": 1},  # bad x
            42,  # not a dict
            {"prompt": "keeper", "x": 0.5, "y": 0, "w": 0.5, "h": 1},
        ]
    )
    assert [r["prompt"] for r in out] == ["keeper"]


def test_normalize_regions_caps_at_max():
    many = [{"prompt": f"r{i}", "x": 0, "y": 0, "w": 1, "h": 1} for i in range(10)]
    out = se.normalize_regions(many)
    assert len(out) == se.REGION_MAX


def test_normalize_regions_none_and_objects():
    assert se.normalize_regions(None) == []
    assert se.normalize_regions([]) == []
    class Obj:
        def model_dump(self):
            return {"prompt": "obj", "x": 0, "y": 0, "w": 1, "h": 1}
    out = se.normalize_regions([Obj()])
    assert out[0]["prompt"] == "obj"
