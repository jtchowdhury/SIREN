"""
finalstate_h5_to_csv.py
=======================
Turn the PYTHIA DIS secondaries HDF5 into a small, human-readable CSV of
final-state hadrons that SIREN's hadronic_showers demo can read directly --
no h5py needed at run time.

NOTE ON NAMING: you asked for a "pkl -> csv" step, but the final-state data is
the PYTHIA output HDF5 (pythia_dis_secondaries.h5), not a pickle. This reads
that h5. If you actually have a *pickle* of final states, tell me its layout
and I'll adapt this.

Input layout (from generate_pythia_dis.py):
  groups  E_nu_1e+02 .. E_nu_1e+05 ; each has, per event, its 20 brightest
  secondaries as  top20_pids  and  top20_energies [GeV].

Output CSV columns:
  event, species, E_GeV        (+ an E_nu_GeV column if --all-enu)

Only species the shower model knows are kept. K0/K0bar (pid 311) is split
50/50 into KS/KL. Leptons/photons/exotics are dropped (not hadrons here).

Run ON THE CLUSTER where the h5 lives, e.g.:
    python finalstate_h5_to_csv.py \
        --h5 /n/home13/jchowdhury/SIREN/resources/analysis/output/pythia_dis_secondaries.h5 \
        --enu 1000 --n-events 20 \
        --out example_final_state.csv
"""
import os
import csv
import argparse

import numpy as np

# pid -> model species name (matches shower_gamma_model.SPECIES)
PID_TO_NAME = {
    111: "pi0", 211: "pip", -211: "pim",
    321: "Kp", -321: "Km", 310: "KS", 130: "KL",
    2212: "p", 2112: "n",
}


def rows_for_group(g, enu_gev, n_events, rng):
    pids_all = np.asarray(g["top20_pids"])
    ens_all = np.asarray(g["top20_energies"])
    n_have = pids_all.shape[0]
    n = n_have if n_events in (None, 0, -1) else min(n_events, n_have)
    rows, skipped = [], set()
    for ev in range(n):
        for pid, E in zip(pids_all[ev], ens_all[ev]):
            pid = int(pid)
            if pid == 0 or not np.isfinite(E) or E <= 0:      # padding
                continue
            name = PID_TO_NAME.get(pid)
            if name is None and abs(pid) == 311:              # K0/K0bar -> KS/KL
                name = "KS" if rng.random() < 0.5 else "KL"
            if name is None:
                skipped.add(pid)
                continue
            rows.append((ev, name, float(E), float(enu_gev)))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="pythia_dis_secondaries.h5")
    ap.add_argument("--out", default="example_final_state.csv")
    ap.add_argument("--enu", type=float, default=1000.0,
                    help="neutrino-energy group [GeV]: 100/1000/1e4/1e5")
    ap.add_argument("--all-enu", action="store_true",
                    help="export every E_nu group (adds an E_nu_GeV column)")
    ap.add_argument("--n-events", type=int, default=20,
                    help="events per group to export (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py
    rng = np.random.default_rng(args.seed)

    all_rows, skipped = [], set()
    with h5py.File(args.h5, "r") as f:
        if args.all_enu:
            groups = sorted(f.keys())
        else:
            key = f"E_nu_{args.enu:.0e}"                       # e.g. E_nu_1e+03
            if key not in f:
                raise SystemExit(f"{key} not in {args.h5}; groups = {list(f.keys())}")
            groups = [key]
        for key in groups:
            # group name E_nu_1e+03 -> 1000.0
            enu = float(key.replace("E_nu_", ""))
            rows, sk = rows_for_group(f[key], enu, args.n_events, rng)
            all_rows.extend(rows)
            skipped |= sk

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        if args.all_enu:
            w.writerow(["event", "species", "E_GeV", "E_nu_GeV"])
            for ev, name, E, enu in all_rows:
                w.writerow([ev, name, f"{E:.6g}", f"{enu:.6g}"])
        else:
            w.writerow(["event", "species", "E_GeV"])
            for ev, name, E, _enu in all_rows:
                w.writerow([ev, name, f"{E:.6g}"])

    if skipped:
        print(f"  (dropped non-hadron/unknown pids: {sorted(skipped)})")
    print(f"wrote {args.out}  ({len(all_rows)} hadron rows)")


if __name__ == "__main__":
    main()
