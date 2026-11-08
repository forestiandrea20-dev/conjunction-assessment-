import pytest
from conjunction.propagation import iso_to_jd, load_omm_csv

def test_iso_to_jd_matches_omm_epoch():
    # The OMM epoch string is converted independently by sgp4.omm;
    # our conversion must agree with it.
    sat = load_omm_csv("data/event_01/gp_data.csv")["STARLINK-30953"]
    jd, fr = iso_to_jd("2026-11-07T04:00:01.999584")
    assert jd == sat.jdsatepoch
    # 1e-10 day is about 9 microseconds: well below the 1 ms resolution we need
    assert fr == pytest.approx(sat.jdsatepochF, abs=1e-10)