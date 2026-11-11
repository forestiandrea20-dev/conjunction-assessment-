"""Close-approach screening and time of closest approach (TCA) refinement."""

import numpy as np
from sgp4.api import Satrec

from conjunction.propagation import propagate


def range_rate(primary: Satrec, secondary: Satrec,
               jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Return r_rel · v_rel [km^2/s] on a time grid.

    Negative while the objects approach, positive while they separate;
    zero crossings from - to + are local minima of the distance.
    """
    # 1. Propagate both objects on the SAME (jd, fr) grid with `propagate`
    (pos_p, vel_p) = propagate(primary, jd, fr)
    (pos_s, vel_s) = propagate(secondary, jd, fr)
    
    # 2. Relative state, convention: secondary - primary
    r_rel = pos_s - pos_p          # (N, 3) km
    v_rel = vel_s - vel_p          # (N, 3) km/s
    # 3. Row-wise dot product -> array of shape (N,)
    return np.sum(r_rel * v_rel, axis=1)
