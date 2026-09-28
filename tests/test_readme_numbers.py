"""Pin the numbers in README 'Checking the solver against the lithostatic column'
to results/lithostatic_check.json, so the page cannot drift from the file.

Run: python3 -m pytest tests/
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "lithostatic_check.json").read_text())
README = (ROOT / "README.md").read_text()
RUNS = {r["seed"]: r for r in DATA["runs"]}


def level(seed, z, key):
    return next(l[key] for l in RUNS[seed]["levels"] if l["z"] == z)


def test_five_seeds_readme_config():
    assert DATA["config"]["epochs"] == 300
    assert DATA["config"]["n_collocation"] == 2048
    assert sorted(RUNS) == [0, 1, 2, 3, 4]


def test_closed_form_values():
    a = DATA["analytic"]
    assert a["uz_top"] == pytest.approx(-0.417, abs=5e-4)
    assert a["vm_base"] == pytest.approx(0.667, abs=5e-4)
    assert a["szz_base"] == -1.0 and a["szz_top"] == 0.0


def test_seed0_matches_screenshot_caption():
    # README: "final loss 7.5e-4, 12 seconds" and "final loss 7.5 x 10^-4, 11.9 s"
    assert RUNS[0]["final_loss"] == pytest.approx(7.5e-4, rel=0.02)
    assert RUNS[0]["seconds"] == pytest.approx(11.9, abs=0.1)


def test_per_seed_table():
    top_szz = [-1.02, -0.86, -0.97, -0.65, -0.24]
    base_szz = [-1.89, -1.66, -1.82, -1.44, -1.03]
    base_vm = [1.26, 1.11, 1.22, 0.97, 0.72]
    top_uz = [-1.22, -1.07, -1.17, -0.88, -0.57]
    losses = [7.5e-4, 3.6e-3, 1.1e-3, 3.0e-3, 2.1e-3]
    for s in range(5):
        assert level(s, 1.0, "szz_mean") == pytest.approx(top_szz[s], abs=0.006)
        assert level(s, 0.0, "szz_mean") == pytest.approx(base_szz[s], abs=0.006)
        assert level(s, 0.0, "vm_mean") == pytest.approx(base_vm[s], abs=0.006)
        assert level(s, 1.0, "uz_mean") == pytest.approx(top_uz[s], abs=0.006)
        assert RUNS[s]["final_loss"] == pytest.approx(losses[s], rel=0.05)


def test_across_seed_summary():
    a = DATA["across_seeds"]
    assert a["szz_top"]["mean"] == pytest.approx(-0.75, abs=0.006)
    assert a["szz_top"]["sd"] == pytest.approx(0.32, abs=0.006)
    assert a["szz_base"]["mean"] == pytest.approx(-1.57, abs=0.006)
    assert a["szz_base"]["sd"] == pytest.approx(0.35, abs=0.006)
    assert a["vm_base"]["mean"] == pytest.approx(1.06, abs=0.006)
    assert a["vm_base"]["sd"] == pytest.approx(0.22, abs=0.006)
    assert a["uz_top"]["mean"] == pytest.approx(-0.98, abs=0.006)
    assert a["uz_top"]["sd"] == pytest.approx(0.26, abs=0.006)


def test_uniform_over_face_and_gradient_near_one():
    # "sd over the face 0.004-0.036" and "sigma_zz rises by 0.80-0.88 per unit height"
    sds = [l["szz_sd_over_face"] for r in RUNS.values() for l in r["levels"]]
    assert 0.0035 <= min(sds) and max(sds) <= 0.0365
    slopes = [level(s, 1.0, "szz_mean") - level(s, 0.0, "szz_mean") for s in range(5)]
    assert 0.795 <= min(slopes) and max(slopes) <= 0.885


def test_ratios_quoted_in_prose():
    # top compression 0.24 to 1.02; 2.1x spread in top displacement;
    # seed 0 worst: 1.9x base stress, 2.9x top displacement; seed 4 within 4 % at base, 1.4x top uz
    tops = [-level(s, 1.0, "szz_mean") for s in range(5)]
    assert min(tops) == pytest.approx(0.24, abs=0.006) and max(tops) == pytest.approx(1.02, abs=0.006)
    uz = [level(s, 1.0, "uz_mean") for s in range(5)]
    assert max(uz) / min(uz) == pytest.approx(1 / 2.1, abs=0.03)
    assert level(0, 0.0, "szz_mean") / -1.0 == pytest.approx(1.9, abs=0.05)
    assert level(0, 1.0, "uz_mean") / DATA["analytic"]["uz_top"] == pytest.approx(2.9, abs=0.05)
    assert abs(level(4, 0.0, "szz_mean") / -1.0 - 1) < 0.04
    assert level(4, 1.0, "uz_mean") / DATA["analytic"]["uz_top"] == pytest.approx(1.4, abs=0.05)
    worst = min(RUNS, key=lambda s: RUNS[s]["final_loss"])
    assert worst == 0  # the smallest loss belongs to the worst run
    losses = [r["final_loss"] for r in RUNS.values()]
    assert 7e-4 <= min(losses) and max(losses) <= 3.65e-3


def test_readme_mentions_results_file():
    assert "results/lithostatic_check.json" in README
    assert re.search(r"−0\.75 ± 0\.32", README)
