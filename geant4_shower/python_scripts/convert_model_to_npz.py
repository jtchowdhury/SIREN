"""
convert_model_to_npz.py
=======================
Convert a trained shower model pickle (shower_model.pkl -> a live
ShowerParamInterpolator) into a PORTABLE, plain-numbers .npz file that
SIREN can load with no dependency on this training code and no pickle.

Why: the .pkl stores live Python/scipy objects stamped with the module they
were built in (shower_gamma_model). Loading it elsewhere needs that module +
a custom unpickler, and it can break across numpy/scipy versions. The .npz
instead stores just the fitted NUMBERS the sampler needs; SIREN rebuilds the
interpolating splines at load time. Nothing about the physics changes.

What is stored (only what SAMPLING needs -- not the heavy per-shower fit pools):
  per species (pid):
    logE        dense log10(E) grid the curves are tabulated on
    z_centers   depth-bin centers [cm]
    pm          p(m) for m=1..Kmax on the grid           shape (n, Kmax)
    logN        log mean Cherenkov yield on the grid      shape (n,)
    logNsig     log-yield spread on the grid              shape (n,)
    m{m}_mean   mean of the (transformed) params per m     shape (n, dim_m)
    m{m}_cov    covariance of those params per m           shape (n, dim_m, dim_m)
  plus a JSON __meta__ blob (species map, m availability, logE ranges, constants).

The m=1_all / valley-off / raw fit pools are intentionally dropped: they are
only used by the performance-comparison scripts, never by the sampler.

Run ON THE CLUSTER (where the pkl and shower_gamma_model.py live), e.g.:
    conda activate siren-dev
    cd geant4_shower/python_scripts
    python convert_model_to_npz.py \
        --pkl ../output/data/shower_model.pkl \
        --out ../output/data/shower_model.npz
"""
import os
import sys
import json
import argparse
import datetime as _dt

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True, help="input shower_model.pkl")
    ap.add_argument("--out", default=None,
                    help="output .npz (default: same path with .npz)")
    ap.add_argument("--src", default=None,
                    help="dir containing shower_gamma_model.py "
                         "(default: this script's directory)")
    ap.add_argument("--n-grid", type=int, default=256,
                    help="points per species to tabulate the smooth curves on "
                         "(the model is a cubic spline over a few energies; 256 "
                         "samples reproduce it to far below physics precision)")
    args = ap.parse_args()

    # import the training module so the pkl can be unpickled
    src = args.src or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, src)
    try:
        import shower_gamma_model as M
    except Exception as e:
        raise SystemExit(f"could not import shower_gamma_model from {src!r}: {e}")

    interp = M.load_model(args.pkl)          # uses its module-remap unpickler
    Kmax = getattr(interp, "Kmax", 3)
    pids = list(interp.pid_models.keys())
    if not pids:
        raise SystemExit("model has no species")

    arrays = {}
    meta = {
        "format": "siren-hadronic-shower-model",
        "version": 1,
        "created_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_pkl": os.path.abspath(args.pkl),
        "Kmax": int(Kmax),
        "n_grid": int(args.n_grid),
        "species": {},
        # provenance only -- NOT used by the sampler (params are fit in depth z [cm]):
        "constants": {
            "X0_ICE_CM": float(getattr(M, "X0_ICE_CM", np.nan)),
            "LAMBDA_LCHAR": float(getattr(M, "LAMBDA_LCHAR", np.nan)),
            "note": "L_char veto was applied at build time; profiles fit in z [cm].",
        },
        "note": "Longitudinal Cherenkov profiles modeled as sums of gamma kernels. "
                "Sample only within each species' logE_range.",
    }

    for pid in pids:
        lo, hi = interp.pid_models[pid]["logE_range"]
        logE = np.linspace(float(lo), float(hi), args.n_grid)
        Egrid = 10.0 ** logE
        name = M.PID_TO_NAME.get(pid, str(pid))

        z = np.asarray(interp.z_centers(pid), float)
        pm = np.array([interp.p_m(pid, E) for E in Egrid], float)          # (n, Kmax)
        logN = np.array([np.log(max(interp.yield_mean(pid, E), 1e-300))
                         for E in Egrid], float)                            # (n,)
        logNsig = np.array([interp.yield_logsigma(pid, E) for E in Egrid], float)

        arrays[f"{pid}|logE"] = logE
        arrays[f"{pid}|z"] = z
        arrays[f"{pid}|pm"] = pm
        arrays[f"{pid}|logN"] = logN
        arrays[f"{pid}|logNsig"] = logNsig

        Emid = 10.0 ** (0.5 * (lo + hi))
        m_avail, dims = [], {}
        for m in range(1, Kmax + 1):
            mc0 = interp.mean_cov(pid, Emid, m)
            if mc0 is None:                    # this species never has m components
                continue
            dim = int(len(mc0[0]))
            means = np.array([interp.mean_cov(pid, E, m)[0] for E in Egrid], float)
            covs = np.array([interp.mean_cov(pid, E, m)[1] for E in Egrid], float)
            arrays[f"{pid}|m{m}|mean"] = means            # (n, dim)
            arrays[f"{pid}|m{m}|cov"] = covs              # (n, dim, dim)
            m_avail.append(m)
            dims[str(m)] = dim

        meta["species"][str(pid)] = {
            "name": name,
            "m_available": m_avail,
            "dims": dims,
            "logE_range": [float(lo), float(hi)],
            "E_range_GeV": [float(10.0 ** lo), float(10.0 ** hi)],
        }
        print(f"  {name:4s} (pid {pid:>5}): m={m_avail}  "
              f"E=[{10**lo:.3g},{10**hi:.3g}] GeV  nz={len(z)}")

    arrays["__meta__"] = np.array(json.dumps(meta))

    out = args.out or (os.path.splitext(args.pkl)[0] + ".npz")
    np.savez_compressed(out, **arrays)
    sz = os.path.getsize(out) / 1024.0
    print(f"\nwrote {out}  ({sz:.1f} KB)  species={len(pids)}")


if __name__ == "__main__":
    main()
