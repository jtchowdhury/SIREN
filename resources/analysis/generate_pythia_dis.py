"""
generate_pythia_dis.py
----------------------
Runs PYTHIA8 CC DIS events (nu_mu + p) at several neutrino energies and
records the hadronic final-state secondaries for each event.

Output: output/pythia_dis_secondaries.h5

Before running, check that pythia8 is available:
    python -c "import pythia8; print(pythia8.__version__)"
If not, try loading the module first:
    module load pythia8   (or whatever your HPC module is named)

Run:
    python generate_pythia_dis.py
"""

import sys
import os
import numpy as np
import h5py

try:
    import pythia8
except ImportError:
    print("ERROR: pythia8 Python module not found.")
    print("Check HPC module availability: module avail pythia")
    sys.exit(1)

# ── Particle ID constants (PDG numbering) ────────────────────────────────────
PI0       = 111
PIP, PIM  = 211, -211
KP,  KM   = 321, -321
KS,  KL   = 310,  130
PRO, NEU  = 2212, 2112
MUP, MUM  = -13,   13

KAON_IDS    = {KP, KM, KS, KL}
NUCLEON_IDS = {PRO, NEU}
LEPTON_IDS  = {13, -13, 14, -14, 12, -12, 16, -16}  # muons + neutrinos = invisible

def cherenkov_weight(pid):
    """
    Approximate fraction of particle energy that becomes detected Cherenkov
    light in ice, relative to an EM shower of the same energy.

    These are rough approximations based on hadronic calorimetry literature
    (Wigmans, Wiebusch). Verify against full simulation before using in production.

    pi0  -> gamma gamma -> pure EM shower -> 1.0
    gamma -> EM shower (e.g. from eta decay if not turned off) -> 1.0
    pi+- -> hadronic shower, ~45% lost to muon decay / nuclear binding -> ~0.55
    K    -> higher muon decay probability than pions -> ~0.50
    p    -> hadronic shower, similar to pi+- -> ~0.50
    n    -> no direct Cherenkov, secondaries only -> ~0.30
    """
    if pid == PI0 or abs(pid) == 22:   # pi0 or photon -> EM
        return 1.0
    elif abs(pid) == 211:
        return 0.55
    elif pid in KAON_IDS:
        return 0.50
    elif pid == PRO:
        return 0.50
    elif pid == NEU:
        return 0.30
    else:
        return 0.50  # default for other hadrons


def run_dis_events(E_nu_GeV, n_events=10000):
    """
    Generate CC DIS events at neutrino energy E_nu_GeV (lab frame).
    Returns a dict of numpy arrays, one entry per accepted event.
    """
    pythia = pythia8.Pythia("", False)  # "" = no config file, False = quiet

    # CC weak boson exchange: nu_mu + p -> mu- + X
    pythia.readString("WeakBosonExchange:ff2ff(t:W) = on")
    pythia.readString("Beams:idA = 14")            # nu_mu
    pythia.readString("Beams:idB = 2212")          # proton (isoscalar approx)
    pythia.readString("Beams:frameType = 2")       # specify eA, eB separately
    pythia.readString(f"Beams:eA = {E_nu_GeV}")    # neutrino energy [GeV]
    pythia.readString("Beams:eB = 0.938272")       # proton at rest [GeV]
    # pi0 decays to gamma gamma with lifetime ~8e-17 s — PYTHIA processes this
    # decay before we see the pi0 as a final-state particle, making f_EM = 0.
    # Keep pi0 stable so it appears in isFinal() and we can count it correctly.
    # Similarly keep eta (221) stable; it also decays fast to photons/pions.
    pythia.readString("111:mayDecay = off")   # pi0 stable: decays to gamma gamma
    # before we can count it. eta (221) left to decay normally so its daughter
    # pions appear in the final state and are counted correctly.
    pythia.readString("Print:quiet = on")
    pythia.readString("Init:showProcesses = off")
    pythia.readString("Init:showChangedSettings = off")

    if not pythia.init():
        raise RuntimeError(f"PYTHIA8 init failed for E_nu = {E_nu_GeV} GeV")

    records = {
        "E_had":             [],   # total hadronic energy [GeV]
        "f_EM":              [],   # EM fraction = E(pi0) / E_had
        "n_total":           [],   # total number of hadronic final-state particles
        "n_pi0":             [],
        "n_pich":            [],   # pi+ + pi-
        "n_kaon":            [],
        "n_nucleon":         [],
        "top20_energies":    [],   # energies of top 20 secondaries, sorted descending
        "top20_pids":        [],   # their PDG IDs
        "top20_cher_weights":[],   # Cherenkov weights for each
    }

    n_done = 0
    while n_done < n_events:
        if not pythia.next():
            continue

        # Collect final-state hadronic particles (skip leptons + neutrinos)
        hadrons = []
        E_had   = 0.0
        E_pi0   = 0.0

        for i in range(pythia.event.size()):
            p = pythia.event[i]
            if not p.isFinal():
                continue
            pid = p.id()
            if abs(pid) in LEPTON_IDS:
                continue
            e = p.e()
            hadrons.append((e, pid))
            E_had += e
            # EM energy: pi0, eta, and any photons that survived (e.g. from
            # other sources). pi0 and eta are kept stable above, so they
            # appear directly here.
            # EM energy: stable pi0 + photons from eta->gamma gamma
            if pid == PI0 or abs(pid) == 22:
                E_pi0 += e

        if E_had < 0.5:   # skip near-elastic events with negligible hadronic energy
            continue

        hadrons.sort(reverse=True)  # sort by energy descending

        f_EM = E_pi0 / E_had

        # Pad to 20 entries so every event has the same shape
        top20 = hadrons[:20]
        pad   = 20 - len(top20)
        top20_e   = [x[0] for x in top20] + [0.0] * pad
        top20_pid = [x[1] for x in top20] + [0]   * pad
        top20_cw  = [cherenkov_weight(x[1]) for x in top20] + [0.0] * pad

        records["E_had"].append(E_had)
        records["f_EM"].append(f_EM)
        records["n_total"].append(len(hadrons))
        records["n_pi0"].append(sum(1 for _, pid in hadrons if pid == PI0))
        records["n_pich"].append(sum(1 for _, pid in hadrons if abs(pid) == 211))
        records["n_kaon"].append(sum(1 for _, pid in hadrons if pid in KAON_IDS))
        records["n_nucleon"].append(sum(1 for _, pid in hadrons if pid in NUCLEON_IDS))
        records["top20_energies"].append(top20_e)
        records["top20_pids"].append(top20_pid)
        records["top20_cher_weights"].append(top20_cw)

        n_done += 1

    pythia.stat()
    return {k: np.array(v) for k, v in records.items()}


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Neutrino energies to scan. E_had = E_nu * y, so each E_nu gives a broad
    # E_had distribution. We store E_had per event and bin in the analysis script.
    E_nu_values = [1e2, 1e3, 1e4, 1e5]  # GeV: 100 GeV, 1 TeV, 10 TeV, 100 TeV
    N_EVENTS    = 10000

    outfile = "output/pythia_dis_secondaries.h5"
    with h5py.File(outfile, "w") as f:
        for E_nu in E_nu_values:
            label = f"E_nu_{E_nu:.0e}"
            print(f"\nGenerating {N_EVENTS} events at E_nu = {E_nu:.0e} GeV ...")
            data = run_dis_events(E_nu, n_events=N_EVENTS)

            grp = f.create_group(label)
            for key, arr in data.items():
                grp.create_dataset(key, data=arr, compression="gzip")

            print(f"  E_had  : {data['E_had'].min():.1f} – {data['E_had'].max():.1f} GeV"
                  f"  (mean {data['E_had'].mean():.1f})")
            print(f"  f_EM   : {data['f_EM'].mean():.3f} ± {data['f_EM'].std():.3f}")
            print(f"  n_total: {data['n_total'].mean():.1f} ± {data['n_total'].std():.1f}")

    print(f"\nSaved to {outfile}")
