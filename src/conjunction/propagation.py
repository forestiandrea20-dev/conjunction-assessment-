# Propagation form a conjunctions event form SOCRATES 
# SOCRATES—Satellite Orbital Conjunction Reports Assessing Threatening Encounters in Space

# Libreries Import
# sgp4
from sgp4 import omm
# sgp4.omm - Support for Orbit Mean-Elements Message (OMM) orbital elements format.
# FUNCTIONS
    #initialize(sat, fields, gravconst=1)

from sgp4.api import Satrec
# help: python -c "from sgp4.api import Satrec; help(Satrec)"


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

