#!/usr/bin/env python3
"""
pythia_h5_to_npz.py  --  flatten the per-energy PYTHIA HDF5 into the single .npz
the shower model and the C++ GammaDISFromSpline read.

generate_pythia_dis.py writes output/pythia_dis_secondaries_nue.h5 with one group
per neutrino energy (E_nu_1e+02, E_nu_1e+03, ...), each holding E_had, top20_pids,
top20_energies, top20_momenta, p_had, p_lepton, etc. This script concatenates all
events across all groups into flat arrays and saves them as pythia_events_nue.npz.

Keys written (dtypes chosen so cnpy can read them in C++):
    E_had            float64  (N,)
    E_nu             float64  (N,)      the generating neutrino energy per event
    top20_pids       int64    (N, 20)
    top20_energies   float64  (N, 20)
    top20_momenta    float64  (N, 20, 4)   [E, px, py, pz] per secondary
    p_had            float64  (N, 4)        total hadronic four-momentum
    p_lepton         float64  (N, 4)        scattered charged-lepton four-momentum

Uses np.savez (uncompressed) so cnpy reads it without inflate.

Usage:
    python pythia_h5_to_npz.py output/pythia_dis_secondaries_nue.h5 \
        ../showers/GammaShowerModel-v1.0/pythia_events_nue.npz
"""
import sys
import numpy as np
import h5py

# datasets to carry through, with the dtype each should end up as
FLOAT_KEYS = ["E_had", "top20_energies", "top20_momenta", "p_had", "p_lepton"]
INT_KEYS   = ["top20_pids"]


def main(src, dst):
    cols = {k: [] for k in FLOAT_KEYS + INT_KEYS}
    e_nu = []

    with h5py.File(src, "r") as f:
        groups = list(f.keys())
        if not groups:
            raise RuntimeError(f"{src} has no groups")
        for label in groups:
            grp = f[label]
            # E_nu for this group, parsed from the label "E_nu_<value>"
            enu_val = float(label.replace("E_nu_", ""))
            n = grp["E_had"].shape[0]
            e_nu.append(np.full(n, enu_val, dtype=np.float64))
            for k in FLOAT_KEYS + INT_KEYS:
                if k not in grp:
                    raise KeyError(f"group {label} missing dataset '{k}' "
                                   f"(regenerate the HDF5 with the updated generator)")
                cols[k].append(np.asarray(grp[k]))

    out = {"E_nu": np.concatenate(e_nu)}
    for k in FLOAT_KEYS:
        out[k] = np.concatenate(cols[k]).astype(np.float64)
    for k in INT_KEYS:
        out[k] = np.concatenate(cols[k]).astype(np.int64)

    np.savez(dst, **out)
    N = out["E_had"].shape[0]
    print(f"wrote {dst}")
    print(f"  events: {N}   from {len(e_nu)} energy groups")
    for k in sorted(out):
        print(f"    {k:16s} {out[k].shape}  {out[k].dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
