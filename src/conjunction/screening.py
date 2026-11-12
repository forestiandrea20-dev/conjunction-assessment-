"""Close-approach screening and time of closest approach (TCA) refinement."""

import numpy as np
from sgp4.api import Satrec
from scipy.optimize import brentq
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


def find_candidates(primary: Satrec, secondary: Satrec,
                    jd0: float, fr0: float, duration_s: float,
                    step_s: float = 60.0) -> list[tuple[float, float]]:
    """Return (t_start, t_end) intervals, in seconds from (jd0, fr0),
    where r_rel · v_rel changes sign from negative to positive."""
    # 1. Time grid t from 0 to duration_s with spacing step_s
    t = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    # 2. fr = fr0 + t / 86400 ; jd = np.full_like(fr, jd0)
    fr = fr0 +t/86400
    jd = np.full_like(fr, jd0)
    # 3. f = range_rate(...)
    f = range_rate(primary, secondary, jd , fr)
    # 4. Indices i where f[i] < 0 and f[i + 1] >= 0
    #    (vectorised: compare f[:-1] and f[1:])
    crossing_indices = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]

    # 5. Return the list of (t[i], t[i + 1]) pairs
    return [(t[i], t[i + 1]) for i in crossing_indices]


def refine_tca(primary: Satrec, secondary: Satrec,
               jd0: float, fr0: float,
               t_start: float, t_end: float) -> tuple[float, float, float]:
    """Return (t_tca [s from (jd0, fr0)], miss distance [km],
    relative speed [km/s]) inside one candidate interval."""

    # Scalar wrapper: brentq passes one time, range_rate needs arrays
    def range_rate_at(t: float) -> float:
        fr = np.array([fr0 + t / 86400.0])
        jd = np.array([jd0])
        f_array = range_rate(primary, secondary, jd, fr)
        return f_array[0]

    # Zero of r_rel · v_rel inside the bracket = time of closest approach
    t_tca = brentq(range_rate_at, t_start, t_end, xtol=1e-6)

    # brentq: Uses the classic Brent’s method to find a root of the function f 
    # on the sign changing interval [a , b].

    # Relative state at the refined TCA
    fr_tca = np.array([fr0 + t_tca / 86400.0])
    jd_tca = np.array([jd0])
    pos_p, vel_p = propagate(primary, jd_tca, fr_tca)
    pos_s, vel_s = propagate(secondary, jd_tca, fr_tca)

    miss_km = np.linalg.norm(pos_s - pos_p, axis=1)[0]
    speed_km_s = np.linalg.norm(vel_s - vel_p, axis=1)[0]

    return float(t_tca), float(miss_km), float(speed_km_s)