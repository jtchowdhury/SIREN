"""
pythia_events_to_npz.py
======================
Dump ALL PYTHIA DIS events from pythia_dis_secondaries.h5 into one compact,
portable .npz so the runtime event-profile plots (hadronic_showers.plot_events*,
plot_event_fluctuation) need only numpy -- no h5py and no h5 file at run time.

Stores, concatenated over every E_nu group:
    E_had           (n_events,)          total hadronic energy [GeV]
    top20_pids      (n_events, 20)        PDG ids of the 20 brightest secondaries
    top20_energies  (n_events, 20)        their energies [GeV]
    E_nu            (n_events,)           the neutrino-energy group each came from

If the HDF5 was made by the updated generator (includes four-momenta), these are
also carried through (needed for momentum-conserving event selection / boosting):
    top20_momenta   (n_events, 20, 4)     [E, px, py, pz] per secondary
    p_had           (n_events, 4)         total hadronic four-momentum
    p_lepton        (n_events, 4)         scattered charged-lepton four-momentum

Run where the h5 lives (e.g. the cluster), then keep the .npz next to your model:
    python pythia_events_to_npz.py \
        --h5  ../../resources/analysis/output/pythia_dis_secondaries.h5 \
        --out ../../resources/analysis/output/pythia_events.npz
"""
import os
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="pythia_dis_secondaries.h5")
    ap.add_argument("--out", default="pythia_events.npz")
    args = ap.parse_args()

    import h5py
    EH, PD, EN, ENU = [], [], [], []
    MOM, PHAD, PLEP = [], [], []       # four-momenta (only if present in the HDF5)
    has_mom = True
    with h5py.File(args.h5, "r") as f:
        for key in sorted(f):
            g = f[key]
            eh = np.asarray(g["E_had"], float)
            EH.append(eh)
            PD.append(np.asarray(g["top20_pids"]))
            EN.append(np.asarray(g["top20_energies"], float))
            try:
                enu = float(key.replace("E_nu_", ""))
            except ValueError:
                enu = np.nan
            ENU.append(np.full(len(eh), enu))
            # carry the four-momenta through if the HDF5 has them; the first group
            # that lacks them disables momentum output for the whole file
            if has_mom and all(k in g for k in ("top20_momenta", "p_had", "p_lepton")):
                MOM.append(np.asarray(g["top20_momenta"], float))
                PHAD.append(np.asarray(g["p_had"], float))
                PLEP.append(np.asarray(g["p_lepton"], float))
            else:
                has_mom = False
            print(f"  {key}: {len(eh)} events")

    EH = np.concatenate(EH)
    PD = np.concatenate(PD, axis=0)
    EN = np.concatenate(EN, axis=0)
    ENU = np.concatenate(ENU)

    out = dict(E_had=EH, top20_pids=PD, top20_energies=EN, E_nu=ENU)
    if has_mom:
        out["top20_momenta"] = np.concatenate(MOM, axis=0)
        out["p_had"]         = np.concatenate(PHAD, axis=0)
        out["p_lepton"]      = np.concatenate(PLEP, axis=0)
    else:
        print("  (no four-momenta in this HDF5 -- regenerate with the updated "
              "generate_pythia_dis.py to include top20_momenta / p_had / p_lepton)")

    np.savez_compressed(args.out, **out)
    sz = os.path.getsize(args.out) / 1024.0
    print(f"\nwrote {args.out}: {len(EH)} events, {PD.shape[1]} secondaries each "
          f"({sz:.1f} KB)")
    print(f"  keys: {sorted(out)}")


if __name__ == "__main__":
    main()
