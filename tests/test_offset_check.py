"""Pin the numbers in README 'The column again, 25 seeds' to
results/offset_check.json (25 seeds at the README configuration; seeds 0-4
are the runs of results/lithostatic_check.json, reproduced).

Run: python3 -m pytest tests/
"""

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "offset_check.json").read_text())
OLD = json.loads((ROOT / "results" / "lithostatic_check.json").read_text())
README = " ".join((ROOT / "README.md").read_text().split())
RUNS = DATA["runs"]
Z = [0.0, 0.25, 0.5, 0.75, 1.0]


def mean_sd(values, ddof=1):
    n = len(values)
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - ddof))
    return m, sd


def spearman(x, y):
    """Rank correlation (no ties expected in these continuous values)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order, 1):
            r[i] = rank
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy)


def fmt(m, sd, nd=2):
    return f"{m:.{nd}f} ± {sd:.{nd}f}".replace("-", "−")


def summary_rows():
    """Rows of the five-seed vs 25-seed table, as printed in the README."""
    off = [r["offset_top"] for r in RUNS]
    base = [r["levels"][0]["szz_mean"] for r in RUNS]
    vm = [r["levels"][0]["vm_mean"] for r in RUNS]
    uz = [r["uz_top"] for r in RUNS]
    grad = [r["szz_gradient"] for r in RUNS]
    o = OLD["across_seeds"]
    rows = [
        ("compression at the top face", fmt(-o["szz_top"]["mean"], o["szz_top"]["sd"]),
         fmt(*mean_sd(off)) + f" ({min(off):.2f} to {max(off):.2f}, median {sorted(off)[12]:.2f})", "0"),
        ("σ_zz at the base", fmt(o["szz_base"]["mean"], o["szz_base"]["sd"]),
         fmt(*mean_sd(base)) + f" ({min(base):.2f} to {max(base):.2f})".replace("-", "−"), "−1"),
        ("von Mises at the base", fmt(o["vm_base"]["mean"], o["vm_base"]["sd"]),
         fmt(*mean_sd(vm)) + f" ({min(vm):.2f} to {max(vm):.2f})", "0.667"),
        ("u_z at the top", fmt(o["uz_top"]["mean"], o["uz_top"]["sd"]),
         fmt(*mean_sd(uz)) + f" ({min(uz):.2f} to {max(uz):.2f})".replace("-", "−"), "−0.417"),
        ("gradient of σ_zz with height", "0.80 to 0.88",
         fmt(*mean_sd(grad)) + f" ({min(grad):.2f} to {max(grad):.2f})", "1"),
    ]
    return [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]


def level_rows():
    rows = []
    for zi, z in enumerate(Z):
        szz = [r["levels"][zi]["szz_mean"] for r in RUNS]
        ratio = [r["sxx_over_szz_levels"][zi] for r in RUNS]
        vmr = [r["vm_over_abs_szz_levels"][zi] for r in RUNS]
        rows.append(
            f"| {z:g} | {fmt(*mean_sd(szz))} | {fmt(*mean_sd(ratio), 3)} "
            f"({min(ratio):.2f} to {max(ratio):.2f}) | {fmt(*mean_sd(vmr), 3)} "
            f"({min(vmr):.2f} to {max(vmr):.2f}) |".replace("-", "−")
        )
    return rows


def test_25_seeds_reproduce_the_five():
    assert DATA["n_seeds"] == 25 and [r["seed"] for r in RUNS] == list(range(25))
    assert DATA["config"]["epochs"] == 300 and DATA["config"]["n_collocation"] == 2048
    for s in range(5):
        new, old = RUNS[s], OLD["runs"][s]
        assert new["final_loss"] == old["final_loss"]
        for a, b in zip(new["levels"], old["levels"]):
            assert a["szz_mean"] == b["szz_mean"] and a["uz_mean"] == b["uz_mean"]
    assert DATA["seconds_total"] == pytest.approx(298, abs=1)
    assert max(r["szz_fit_max_resid"] for r in RUNS) < 0.0085  # "largest residual 0.008"


def test_every_seed_compressive_and_the_five_seed_spread():
    off = [r["offset_top"] for r in RUNS]
    assert DATA["n_offset_compressive"] == 25 and min(off) > 0
    assert sorted(off)[:3] == pytest.approx([0.232, 0.245, 0.410], abs=5e-4)
    assert [r["seed"] for r in sorted(RUNS, key=lambda r: r["offset_top"])][:3] == [4, 10, 8]
    assert sum(0.635 <= v <= 1.015 for v in off) == 22
    m, sd = mean_sd(off)
    assert (m, sd) == pytest.approx((0.77, 0.21), abs=5e-3)
    assert mean_sd(off[:5])[1] == pytest.approx(0.32, abs=5e-3)


def test_summary_table_rows_in_readme():
    for row in summary_rows():
        assert row in README, row


def test_level_table_rows_in_readme():
    for row in level_rows():
        assert row in README, row


def test_overburden_prediction():
    # u_z at the top from the fitted stress profile, under lateral confinement
    rel = [r["uz_pred_rel_err_top"] for r in RUNS]
    m, sd = mean_sd(rel)
    assert (m, sd) == pytest.approx((0.024, 0.017), abs=5e-4)
    assert (min(rel), max(rel)) == pytest.approx((0.006, 0.083), abs=5e-4)
    share_off = [r["uz_top_excess_from_offset"] / r["uz_top_excess"] for r in RUNS]
    share_all = [r["uz_top_excess_predicted"] / r["uz_top_excess"] for r in RUNS]
    assert mean_sd(share_off) == pytest.approx((1.10, 0.05), abs=5e-3)
    assert (min(share_off), max(share_off)) == pytest.approx((1.05, 1.26), abs=5e-3)
    assert mean_sd(share_all) == pytest.approx((0.95, 0.06), abs=5e-3)
    assert (min(share_all), max(share_all)) == pytest.approx((0.71, 0.99), abs=5e-3)
    grad = [r["szz_gradient"] for r in RUNS]
    assert (min(grad), max(grad)) == pytest.approx((0.787, 0.875), abs=5e-4)
    assert max(grad) < 0.88  # "never 1.0": 12-21 % short in every seed


def test_confinement_by_level_and_lateral_displacement():
    base = [r["sxx_over_szz_levels"][0] for r in RUNS]
    assert mean_sd(base) == pytest.approx((0.332, 0.001), abs=5e-4)
    top = [r["sxx_over_szz_levels"][4] for r in RUNS]
    assert mean_sd(top) == pytest.approx((0.21, 0.13), abs=5e-3)
    vm_top = [r["vm_over_abs_szz_levels"][4] for r in RUNS]
    assert (min(vm_top), max(vm_top)) == pytest.approx((0.71, 1.49), abs=5e-3)
    lat = [max(r["ux_abs_mean_max"], r["uy_abs_mean_max"]) / abs(r["uz_top"]) for r in RUNS]
    assert (min(lat), max(lat)) == pytest.approx((0.012, 0.359), abs=5e-4)
    assert sum(v > 0.10 for v in lat) == 12 and sum(v > 0.20 for v in lat) == 5
    # the three smallest offsets are the three least confined columns
    by_seed = {r["seed"]: r for r in RUNS}
    for seed, l, ratio, vm in [(4, 0.26, -0.28, 1.49), (10, 0.36, -0.02, 1.36), (8, 0.27, 0.01, 1.01)]:
        r = by_seed[seed]
        assert max(r["ux_abs_mean_max"], r["uy_abs_mean_max"]) / abs(r["uz_top"]) == pytest.approx(l, abs=5e-3)
        assert r["sxx_over_szz_levels"][4] == pytest.approx(ratio, abs=5e-3)
        assert r["vm_over_abs_szz_levels"][4] == pytest.approx(vm, abs=5e-3)


def test_loss_ranks_the_runs_backwards():
    s = DATA["spearman_loss_vs_offset"]
    assert s["rho"] == pytest.approx(-0.69, abs=5e-3) and s["p"] < 2e-4
    s = DATA["spearman_loss_vs_uz_top_error"]
    assert s["rho"] == pytest.approx(-0.72, abs=5e-3) and s["p"] < 1e-4
    ranked = sorted(RUNS, key=lambda r: r["final_loss"])
    low = [r["offset_top"] for r in ranked[:5]]
    high = [r["offset_top"] for r in ranked[-5:]]
    assert (min(low), max(low)) == pytest.approx((0.84, 1.01), abs=5e-3)
    assert sum(low) / 5 == pytest.approx(0.94, abs=5e-3)
    assert (min(high), max(high)) == pytest.approx((0.41, 0.85), abs=5e-3)
    assert sum(high) / 5 == pytest.approx(0.64, abs=5e-3)
    lat = [max(r["ux_abs_mean_max"], r["uy_abs_mean_max"]) / abs(r["uz_top"]) for r in RUNS]
    assert spearman(lat, [r["offset_top"] for r in RUNS]) == pytest.approx(-0.73, abs=5e-3)
    assert spearman(lat, [r["final_loss"] for r in RUNS]) == pytest.approx(0.46, abs=5e-3)
    assert spearman(lat, [r["uz_pred_rel_err_top"] for r in RUNS]) == pytest.approx(0.59, abs=5e-3)
    stats = pytest.importorskip("scipy.stats")  # the two p-values quoted for these
    assert stats.spearmanr(lat, [r["final_loss"] for r in RUNS]).pvalue < 0.025
    assert stats.spearmanr(lat, [r["uz_pred_rel_err_top"] for r in RUNS]).pvalue < 0.0025


def test_readme_mentions_results_file():
    assert "results/offset_check.json" in README
    assert "25 of 25" in README


if __name__ == "__main__":
    print("\n".join(summary_rows()))
    print()
    print("\n".join(level_rows()))
