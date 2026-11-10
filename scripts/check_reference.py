import json
from pathlib import Path 

import numpy as np

from conjunction.propagation import iso_to_jd, load_omm_csv, propagate

# Repository root: absolute path of this script, two levels up (scripts/ -> root),
# so data files are found regardless of the directory the script is run from.
ROOT = Path(__file__).resolve().parents[1]




def check_event(event_dir: Path) -> None:
    #print("checking", event_dir)
    # 1. leggere reference.json (json.load) e caricare gp_data.csv
    ref_path = event_dir / "reference.json"
    gp_path = event_dir / "gp_data.csv"

    with open(ref_path, 'r') as ref_file:
        data_ref = json.load(ref_file)

    sats = load_omm_csv(gp_path)

    #Check load correctly
    # print(data_ref["source"])
    # print(list(sats.keys()))
    #primary_name = data_ref["primary"]["name"]
    #print(primary_name, sats[primary_name].satnum, data_ref["primary"]["norad_id"])


        # Data structures:
    #   data_ref : dict parsed from reference.json (MATLAB struct analogue);
    #              data_ref["primary"] and data_ref["secondary"] are nested dicts
    #              with keys "norad_id" and "name".
    #   sats     : dict {OBJECT_NAME: Satrec} built by load_omm_csv;
    #              the object names stored in data_ref are its keys.
    # 2. Retrieve the two Satrec objects by name

    primary = sats[data_ref["primary"]["name"]]
    #              └──── "STARLINK-30953" ────┘
    #         sats["STARLINK-30953"]  ->  Satrec of Starlink
    secondary = sats[data_ref["secondary"]["name"]]

    # 3. Convert the TCA to a split Julian date
    jd_tca, fr_tca = iso_to_jd(data_ref["tca_utc"])
    # 4. propagare entrambi alla TCA
    (pos_p, vel_p) = propagate(primary, jd_tca, fr_tca)
    (pos_s, vel_s) = propagate(secondary, jd_tca, fr_tca)
        # 5. Relative distance [m] and relative speed [km/s] at the TCA
    dist_rel_m = np.linalg.norm(pos_p - pos_s, axis=1)[0] * 1e3
    #print(dist_rel_m)
   
    vel_rel_kms = np.linalg.norm(vel_p - vel_s, axis=1)[0]
    #print(vel_rel_kms)
    print(f"Conjunction event: {data_ref["primary"]["name"]} - {data_ref["secondary"]["name"]}")
    print(f"Relative distance: {dist_rel_m:8.2f}m.\nRelative velocity {vel_rel_kms:8.2f} km/s")
    # 6. stampare i valori calcolati accanto a quelli di riferimento
    


if __name__ == "__main__":
    for event in ["event_01", "event_02"]:
        check_event(ROOT / "data" / event)