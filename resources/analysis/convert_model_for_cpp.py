#!/usr/bin/env python3
"""
convert_model_for_cpp.py  --  repack shower_model.npz into a NUMERIC-ONLY npz that
the C++ GammaDISFromSpline can read with cnpy.

The Python model file stores its index as a JSON string in the __meta__ array (a
unicode numpy array). cnpy reads numeric arrays, not unicode/JSON, so we drop
__meta__ and add plain integer index arrays instead:

    pids           int64 [P]          the species PDG codes present
    Kmax           int64 [1]
    <pid>|m_avail  int64 [.]          which m have fitted params, per species

Every other array in the original file (<pid>|logE, |z, |pm, |logN, |logNsig,
|m<m>|mean, |m<m>|cov) is copied through unchanged. The result, shower_model_cpp.npz,
is what you pass to GammaDISFromSpline's shower_model_path.

Usage:
    python convert_model_for_cpp.py \
        resources/showers/GammaShowerModel-v1.0/shower_model.npz \
        resources/showers/GammaShowerModel-v1.0/shower_model_cpp.npz
"""
import sys
import json
import numpy as np


def main(src, dst):
    d = np.load(src, allow_pickle=False)
    meta = json.loads(str(d["__meta__"]))

    out = {}
    pids = []
    for pid_str, sp in meta["species"].items():
        pid = int(pid_str)
        pids.append(pid)
        # copy the per-species numeric arrays straight through
        for suffix in ("logE", "z", "pm", "logN", "logNsig"):
            out[f"{pid}|{suffix}"] = np.asarray(d[f"{pid}|{suffix}"], dtype=np.float64)
        m_avail = [int(m) for m in sp["m_available"]]
        out[f"{pid}|m_avail"] = np.asarray(m_avail, dtype=np.int64)
        for m in m_avail:
            out[f"{pid}|m{m}|mean"] = np.asarray(d[f"{pid}|m{m}|mean"], dtype=np.float64)
            out[f"{pid}|m{m}|cov"] = np.asarray(d[f"{pid}|m{m}|cov"], dtype=np.float64)

    out["pids"] = np.asarray(sorted(pids), dtype=np.int64)
    out["Kmax"] = np.asarray([int(meta.get("Kmax", 3))], dtype=np.int64)

    np.savez(dst, **out)
    print(f"wrote {dst}")
    print(f"  species (pids): {out['pids'].tolist()}")
    print(f"  Kmax = {out['Kmax'][0]}")
    print(f"  arrays: {len(out)}   (dropped __meta__ JSON)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
