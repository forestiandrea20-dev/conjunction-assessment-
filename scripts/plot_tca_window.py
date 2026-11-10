"""Plot the relative distance around the SOCRATES TCA for each reference event."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write image files only, no window (works in WSL without a display)
import matplotlib.pyplot as plt
import numpy as np

from conjunction.propagation import iso_to_jd, load_omm_csv, propagate

ROOT = Path(__file__).resolve().parents[1]


def relative_distance_km(primary, secondary, jd_tca, fr_tca, dt_s):
    """Relative distance [km] at times dt_s [s] from the TCA.

    jd stays fixed and only fr varies, so millisecond resolution
    is not lost in the large Julian date.
    """
    fr = fr_tca + dt_s / 86400.0
    jd = np.full_like(fr, jd_tca)
    r_p, _ = propagate(primary, jd, fr)
    r_s, _ = propagate(secondary, jd, fr)
    return np.linalg.norm(r_p - r_s, axis=1)


def plot_event(event_dir: Path, out_dir: Path) -> Path:
    with open(event_dir / "reference.json") as f:
        ref = json.load(f)
    sats = load_omm_csv(event_dir / "gp_data.csv")
    primary = sats[ref["primary"]["name"]]
    secondary = sats[ref["secondary"]["name"]]
    jd_tca, fr_tca = iso_to_jd(ref["tca_utc"])

    # Wide view: +/- 60 s, 0.1 s step
    dt_wide = np.arange(-60.0, 60.0 + 1e-9, 0.1)
    d_wide = relative_distance_km(primary, secondary, jd_tca, fr_tca, dt_wide)
    # Zoom: +/- 50 ms, 0.1 ms step. At ~10 km/s the objects close ~1 m every 0.1 ms,
    # so only a fine grid resolves the rounded bottom of the curve.
    dt_zoom = np.arange(-0.05, 0.05 + 1e-12, 1e-4)
    d_zoom_m = relative_distance_km(primary, secondary, jd_tca, fr_tca, dt_zoom) * 1e3
    i_min = np.argmin(d_zoom_m)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dt_wide, d_wide, lw=1.5, color="C0")
    ax.set_xlabel(f"Time from SOCRATES TCA ({ref['tca_utc']} UTC) [s]")
    ax.set_ylabel("Relative distance [km]")
    ax.set_title(f"{ref['primary']['name']} vs {ref['secondary']['name']} "
                 f"(v_rel = {ref['relative_speed_km_s']:.3f} km/s)")
    ax.grid(alpha=0.3)

    ins = ax.inset_axes([0.58, 0.45, 0.38, 0.45])
    ins.plot(dt_zoom * 1e3, d_zoom_m, lw=1.2, color="C0", label="SGP4, this work")
    ins.axhline(ref["min_range_km"] * 1e3, ls="--", color="0.5",
                label=f"SOCRATES: {ref['min_range_km'] * 1e3:.0f} m")
    ins.plot(dt_zoom[i_min] * 1e3, d_zoom_m[i_min], "o", color="C3", ms=4,
             label=f"min: {d_zoom_m[i_min]:.1f} m at {dt_zoom[i_min] * 1e3:+.1f} ms")
    ins.set_xlabel("Time from TCA [ms]", fontsize=8)
    ins.set_ylabel("[m]", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.grid(alpha=0.3)
    ins.legend(fontsize=7, loc="upper center")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_dir.name}_tca_window.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for event in ["event_01", "event_02"]:
        print("Saved", plot_event(ROOT / "data" / event, ROOT / "results"))