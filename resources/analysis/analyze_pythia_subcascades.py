"""
analyze_pythia_subcascades.py
------------------------------
Analyzes the sub-cascade energy distribution for the E_nu = 100 TeV Pythia
DIS group. Purpose: determine how many sub-cascades exceed our G4 coverage
limit (30 TeV), so we can assess whether the extrapolation in analyze_g4_profiles.py
is reliable.

Run on the cluster from resources/analysis/:
    python analyze_pythia_subcascades.py

Reads: output/pythia_dis_secondaries.h5
"""

import numpy as np
import h5py
import os

INFILE    = "output/pythia_dis_secondaries.h5"
G4_MAX_GEV = 30_000.0   # GeV — top of G4 simulation grid

TARGET_GROUPS = ["E_nu_1e+05", "E_nu_1e+04", "E_nu_1e+03"]   # 100 TeV, 10 TeV, 1 TeV

def nice_label(grp_name):
    e = float(grp_name.replace("E_nu_", ""))
    if e >= 1e6: return f"{e/1e6:.0f} PeV"
    if e >= 1e3: return f"{e/1e3:.0f} TeV"
    return f"{e:.0f} GeV"


def analyze_group(grp, grp_name):
    top20_e    = grp["top20_energies"][:]   # (n_events, 20), total energy p.e() [GeV]
    top20_pids = grp["top20_pids"][:]       # (n_events, 20)
    E_had      = grp["E_had"][:]            # (n_events,)
    n_events   = top20_e.shape[0]

    # --- top-1 (highest-energy) sub-cascade per event ---
    top1_e = top20_e[:, 0]

    # --- all non-padding entries (pid != 0) ---
    mask_nonpad = top20_pids != 0
    all_e       = top20_e[mask_nonpad]      # flattened energies of real secondaries

    print(f"\n{'='*60}")
    print(f"  E_nu = {nice_label(grp_name)}   ({n_events} events)")
    print(f"{'='*60}")

    print(f"\n  E_had (total hadronic energy):")
    print(f"    mean  = {E_had.mean():.1f} GeV   median = {np.median(E_had):.1f} GeV")
    print(f"    range = {E_had.min():.1f} – {E_had.max():.1f} GeV")

    print(f"\n  Top-1 sub-cascade energy (highest-energy secondary per event):")
    print(f"    mean   = {top1_e.mean():.1f} GeV")
    print(f"    median = {np.median(top1_e):.1f} GeV")
    print(f"    max    = {top1_e.max():.1f} GeV  ({top1_e.max()/1000:.1f} TeV)")
    print(f"    > G4 max ({G4_MAX_GEV/1000:.0f} TeV): "
          f"{(top1_e > G4_MAX_GEV).sum()} / {n_events} events "
          f"({100*(top1_e > G4_MAX_GEV).mean():.1f}%)")

    # --- per slot: how often does the k-th secondary exceed G4 max? ---
    print(f"\n  Fraction of events where k-th secondary > {G4_MAX_GEV/1000:.0f} TeV:")
    for k in range(min(10, top20_e.shape[1])):
        ek = top20_e[:, k]
        frac = (ek > G4_MAX_GEV).mean()
        if frac < 0.001 and k > 0:
            print(f"    k={k+1:2d}: {frac*100:.2f}%  (negligible beyond this)")
            break
        print(f"    k={k+1:2d}: {frac*100:.2f}%")

    # --- all secondaries: what fraction exceed G4 max? ---
    print(f"\n  All non-padding secondaries (across all 20 slots, all events):")
    print(f"    total entries = {len(all_e)}")
    print(f"    > {G4_MAX_GEV/1000:.0f} TeV: {(all_e > G4_MAX_GEV).sum()} "
          f"({100*(all_e > G4_MAX_GEV).mean():.2f}%)")
    print(f"    > 10 TeV:    {(all_e > 10000).sum()} "
          f"({100*(all_e > 10000).mean():.2f}%)")
    print(f"    > 1 TeV:     {(all_e > 1000).sum()} "
          f"({100*(all_e > 1000).mean():.2f}%)")

    # --- percentiles of top-1 ---
    pcts = [50, 75, 90, 95, 99]
    print(f"\n  Top-1 sub-cascade energy percentiles:")
    for p in pcts:
        print(f"    p{p:2d} = {np.percentile(top1_e, p):.1f} GeV")


if __name__ == "__main__":
    if not os.path.exists(INFILE):
        raise SystemExit(f"ERROR: {INFILE} not found. Run generate_pythia_dis.py first.")

    with h5py.File(INFILE, "r") as f:
        available = sorted(f.keys())
        print(f"Available groups: {available}")

        for grp_name in TARGET_GROUPS:
            if grp_name not in f:
                print(f"\n[SKIP] {grp_name} not found in file.")
                continue
            analyze_group(f[grp_name], grp_name)

    print("\nDone.")
