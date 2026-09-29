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
            print(f"  {key}: {len(eh)} events")

    EH = np.concatenate(EH)
    PD = np.concatenate(PD, axis=0)
    EN = np.concatenate(EN, axis=0)
    ENU = np.concatenate(ENU)

    np.savez_compressed(args.out, E_had=EH, top20_pids=PD,
                        top20_energies=EN, E_nu=ENU)
    sz = os.path.getsize(args.out) / 1024.0
    print(f"\nwrote {args.out}: {len(EH)} events, {PD.shape[1]} secondaries each "
          f"({sz:.1f} KB)")


if __name__ == "__main__":
    main()
