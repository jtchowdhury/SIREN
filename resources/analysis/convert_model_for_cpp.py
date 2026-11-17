#!/usr/bin/env python3
"""
convert_model_for_cpp.py -- write the shower model as the JSON file that
GammaDISFromSpline reads (through cereal).

Input: shower_model.npz (the Python model) or the older shower_model_cpp.npz.
Output layout:

    {"Kmax": K,
     "species": [{"pid": 211,
                  "logE": [L], "pm": [L*K], "logN": [L], "logNsig": [L],
                  "m": [1, 2],
                  "mean": [[L*(3m-1)] for each m],
                  "cov":  [[L*(3m-1)^2] for each m]}, ...]}

2-d and 3-d arrays are flattened row-major.

Usage:
    python convert_model_for_cpp.py \
        resources/showers/GammaShowerModel-v1.0/shower_model.npz \
        resources/showers/GammaShowerModel-v1.0/shower_model.json
"""
import sys
import json
import numpy as np


def main(src, dst):
    d = np.load(src, allow_pickle=False)
    if "__meta__" in d.files:                       # Python model file
        meta = json.loads(str(d["__meta__"]))
        kmax = int(meta.get("Kmax", 3))
        m_avail = {int(p): [int(m) for m in sp["m_available"]]
                   for p, sp in meta["species"].items()}
    else:                                           # numeric-only npz
        kmax = int(d["Kmax"][0])
        m_avail = {int(p): [int(m) for m in d[f"{int(p)}|m_avail"]] for p in d["pids"]}

    species = []
    for pid in sorted(m_avail):
        def arr(name):
            a = np.asarray(d[f"{pid}|{name}"], dtype=float).ravel()
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{pid}|{name} has non-finite values, which JSON cannot store")
            return a.tolist()
        ms = m_avail[pid]
        species.append({
            "pid": pid,
            "logE": arr("logE"), "pm": arr("pm"), "logN": arr("logN"), "logNsig": arr("logNsig"),
            "m": ms,
            "mean": [arr(f"m{m}|mean") for m in ms],
            "cov": [arr(f"m{m}|cov") for m in ms],
        })

    with open(dst, "w") as f:
        json.dump({"Kmax": kmax, "species": species}, f)
    print(f"wrote {dst}: Kmax = {kmax}, {len(species)} species")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
