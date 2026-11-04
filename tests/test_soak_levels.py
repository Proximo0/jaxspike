"""bench/soak_levels.curve_table: the leveling-curve table (ro5-balance all-bot median column)."""
import numpy as np

from bench.soak_levels import curve_table


def test_curve_table_all_bot_median(capsys):
    first = {10: np.array([100, 200, 300, -1]), 20: np.array([500, -1, -1, -1]), 30: np.full(4, -1)}
    curve_table(first, 1000, 4.0, 0.3)
    out = capsys.readouterr().out.splitlines()
    row10 = next(r for r in out if r.strip().startswith("10 |"))
    row20 = next(r for r in out if r.strip().startswith("20 |"))
    cols = [c.strip() for c in row10.split("|")]
    assert cols[3] == "200" and cols[5] == "250"  # median of the 3 there; all-bot median counts the 4th as not there
    assert [c.strip() for c in row20.split("|")][5] == "> 1000"  # 1 of 4: the all-bot median is not known yet
    assert any("30 |" in r and "not reached" in r for r in out)
