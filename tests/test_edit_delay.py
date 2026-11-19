"""The edit-delay measurement: first edit per perimeter (not later re-edits), bare dates left out, the stamp field
the catalog would use (datecurrent, else gdb_from_date)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import measure_edit_delay as M  # noqa: E402


def ms(s):
    return int(pd.Timestamp(s, tz="UTC").value // 10 ** 6)


def test_first_edit_per_perimeter():
    df = pd.DataFrame({
        "sourceglobalid": ["A", "A", "B", "C", None],
        "globalid": ["a1", "a2", "b", "c", "d"],
        "perimeterdatetime": [ms("2024-08-10 22:00"), ms("2024-08-10 22:00"), "2024-08-11 21:30:00", ms("2024-08-12 00:00"), ms("2024-08-12 20:00")],
        "datecurrent": [ms("2024-08-11 03:00"), ms("2024-08-12 18:00"), None, ms("2024-08-12 05:00"), ms("2024-08-13 08:00")],
        "gdb_from_date": [None, None, ms("2024-08-12 06:30"), None, ms("2024-08-13 02:00")],
        "source": ["nifs"] * 5,
    })
    g = M.edit_delays(df).set_index("id")
    assert np.isclose(g.loc["A", "delay_h"], 5.0)            # first edit, not the re-edit 44 h later
    assert np.isclose(g.loc["B", "delay_h"], 9.0)            # string mapping time; no datecurrent -> gdb_from_date
    assert "C" not in g.index                                 # bare date: not a real mapping time
    assert np.isclose(g.loc["d", "delay_h"], 12.0)           # datecurrent preferred (the catalog's stamp order)
    assert np.isclose(g.loc["d", "delay_gdb_from_date_h"], 6.0)
