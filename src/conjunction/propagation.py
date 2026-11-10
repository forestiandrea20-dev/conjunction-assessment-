# Propagation form a conjunctions event form SOCRATES 
# SOCRATES—Satellite Orbital Conjunction Reports Assessing Threatening Encounters in Space

# Libreries Import
# First function 
# sgp4
from sgp4 import omm
# sgp4.omm - Support for Orbit Mean-Elements Message (OMM) orbital elements format.
# FUNCTIONS
    #initialize(sat, fields, gravconst=1)

from sgp4.api import Satrec
# help: python -c "from sgp4.api import Satrec; help(Satrec)"

# Second function 
from datetime import datetime
from sgp4.api import jday

#third 
import numpy as np

# Function - read OMM csv
def load_omm_csv(path):
    """Read an OMM CSV file and return {object name: Satrec}."""
    sats = {}
    with open(path) as f:          
        for fields in omm.parse_csv(f):
            sat = Satrec()
            omm.initialize(sat, fields)
            sats[fields["OBJECT_NAME"]] = sat
    return sats


if __name__ == "__main__":
    sats = load_omm_csv("data/event_01/gp_data.csv")
    print(sats.keys())

# Function - from ISO string to (jd, fr) 
def iso_to_jd(iso: str) -> tuple[float, float]:
    """Convert an ISO 8601 UTC string to a split Julian date (jd, fr) for SGP4."""
    # example of date "2026-11-09T08:56:34.405"
    dt = datetime.fromisoformat(iso) #naive -> no time zone 

    # NOTE: datetime splits the time into 7 fields (seconds and microseconds are separate),
    # while jday() takes 6 arguments, so microseconds are folded into fractional seconds
    # # to preserve millisecond precision of the TCA.

    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second+dt.microsecond*1e-6)

    return jd, fr

def propagate(sat, jd, fr):
    """Propagate a Satrec to the given split Julian dates.

    Parameters
    ----------
    sat : Satrec
    jd, fr : array_like
        Integer and fractional parts of the Julian dates (UTC).

    Returns
    -------
    r, v : ndarray, shape (N, 3)
        Position [km] and velocity [km/s] in the TEME frame.
    """
    err, r, v = sat.sgp4_array(np.atleast_1d(jd).astype(float), np.atleast_1d(fr).astype(float))
    
    if np.any(err != 0):
        raise RuntimeError(f"SGP4 error codes: {np.unique(err)}")

    return r, v
