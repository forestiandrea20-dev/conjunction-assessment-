"""Plot r_rel · v_rel around the SOCRATES TCA of one event."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conjunction.propagation import iso_to_jd, load_event
from conjunction.screening import range_rate



def main() -> None:
    event_dir = Path(__file__).resolve().parents[1] / "data" / "event_01"
    primary, secondary, reference = load_event(event_dir)



    # 2. jd_tca, fr_tca = iso_to_jd(reference["tca_utc"])
    jd_tca, fr_tca = iso_to_jd(reference["tca_utc"])
    
    # 3. Time grid: t = np.arange(-600, 601, 1.0) seconds
    t = np.arange(-600, 601, 1.0) # seconds 
    fr = fr_tca + t / 86400.0
    jd = np.full_like(fr, jd_tca)
    # 4. f = range_rate(primary, secondary, jd, fr)
    f = range_rate(primary, secondary, jd, fr)
    print(t.shape, f.shape, f[600])

    t_zoom = np.linspace(-0.1, 0.1, 2001)
    fr_zoom = fr_tca +t_zoom / 86400.0
    jd_zoom = np.full_like(fr_zoom, jd_tca)
    f_zoom = range_rate(primary, secondary, jd_zoom, fr_zoom)
    # 5. Two subplots: f(t) over ±600 s, and a zoom over ±0.1 s
    #    (second grid with sub-millisecond spacing), horizontal line at zero
    
    # Linear fit near TCA: f(t) ≈ v_rel² (t - t_tca), so the intercept
    # over the slope gives the offset of the true TCA from SOCRATES
    slope, intercept = np.polyfit(t_zoom, f_zoom, 1)
    v_rel_ref = reference["relative_speed_km_s"]
    tca_offset_ms = -intercept / slope * 1e3

    print(f"f(0)            = {f[600]:+.6f} km^2/s")
    print(f"fitted slope    = {slope:.3f} km^2/s^2")
    print(f"v_rel^2 (ref)   = {v_rel_ref**2:.3f} km^2/s^2")
    print(f"TCA offset      = {tca_offset_ms:+.4f} ms")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))

    ax_full.plot(t, f)
    ax_full.axhline(0.0, color="k", linewidth=0.8)
    ax_full.set_xlabel("t − TCA [s]")
    ax_full.set_ylabel("r_rel · v_rel [km²/s]")
    ax_full.set_title("±10 min around SOCRATES TCA")
    ax_full.grid(True, alpha=0.3)

    ax_zoom.plot(t_zoom * 1e3, f_zoom, label="SGP4")
    ax_zoom.plot(t_zoom * 1e3, slope * t_zoom + intercept, "--",
                 label=f"linear fit, slope {slope:.2f} km²/s²")
    ax_zoom.axhline(0.0, color="k", linewidth=0.8)
    ax_zoom.axvline(tca_offset_ms, color="r", linewidth=0.8,
                    label=f"zero crossing {tca_offset_ms:+.3f} ms")
    ax_zoom.set_xlabel("t − TCA [ms]")
    ax_zoom.set_ylabel("r_rel · v_rel [km²/s]")
    ax_zoom.set_title("Zoom ±100 ms")
    ax_zoom.legend(fontsize=8)
    ax_zoom.grid(True, alpha=0.3)

    fig.suptitle(f"{reference['primary']['name']} vs "
                 f"{reference['secondary']['name']}")
    fig.tight_layout()

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{event_dir.name}_range_rate.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")

if __name__ == "__main__":
    main()