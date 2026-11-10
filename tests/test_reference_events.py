"""Check that SGP4 propagation reproduces the CelesTrak SOCRATES reference values."""
import json
from pathlib import Path

import numpy as np
import pytest

from conjunction.propagation import iso_to_jd, load_omm_csv, propagate

DATA = Path(__file__).resolve().parents[1] / "data"
EVENTS = sorted(p.name for p in DATA.iterdir() if (p / "reference.json").exists())


def relative_state_at_tca(event_dir: Path):
    """Return (reference dict, distance [m], relative speed [km/s]) at the SOCRATES TCA."""
    with open(event_dir / "reference.json") as f:
        ref = json.load(f)
    sats = load_omm_csv(event_dir / "gp_data.csv")
    jd, fr = iso_to_jd(ref["tca_utc"])
    r_p, v_p = propagate(sats[ref["primary"]["name"]], np.array([jd]), np.array([fr]))
    r_s, v_s = propagate(sats[ref["secondary"]["name"]], np.array([jd]), np.array([fr]))
    dist_m = np.linalg.norm(r_p - r_s, axis=1)[0] * 1e3
    speed_kms = np.linalg.norm(v_p - v_s, axis=1)[0]
    return ref, dist_m, speed_kms


@pytest.mark.parametrize("event", EVENTS)
def test_norad_ids_match_reference(event):
    with open(DATA / event / "reference.json") as f:
        ref = json.load(f)
    sats = load_omm_csv(DATA / event / "gp_data.csv")
    for role in ("primary", "secondary"):
        assert sats[ref[role]["name"]].satnum == ref[role]["norad_id"]


@pytest.mark.parametrize("event", EVENTS)
def test_miss_distance_and_speed_match_socrates(event):
    ref, dist_m, speed_kms = relative_state_at_tca(DATA / event)
    # SOCRATES publishes range and speed with 3 decimals (1 m, 1 m/s):
    # tolerances equal half the last published digit.
    assert dist_m == pytest.approx(ref["min_range_km"] * 1e3, abs=0.5)
    assert speed_kms == pytest.approx(ref["relative_speed_km_s"], abs=5e-4)