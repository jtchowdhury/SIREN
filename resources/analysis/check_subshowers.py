#!/usr/bin/env python3
"""
check_subshowers.py  --  sanity-check the gamma_subshowers output in a SIREN
HDF5 event file.

Each interaction's hadronic blob is developed (by hadronic_showers.BlobDeveloper,
inside _util.SaveEvents) into a flat list of gamma subshowers, stored as the
record field `gamma_subshowers` with four parallel columns:

    subshower_parent      (str)    e.g. "pi0(1)", "Kp(1)"  (m>=2 share one tag)
    subshower_alpha       (float)  gamma shape   (> 0)
    subshower_beta        (float)  gamma rate    (> 0)
    subshower_amplitude   (float)  integrated light of the component  (> 0)

This script flattens those across all events/interactions and checks they are
finite, positive, and structurally sane, then prints a few summary statistics.

Usage:
    python check_subshowers.py output/IceCube.hdf5
    python check_subshowers.py output/IceCube.hdf5 --plot
"""

import argparse
import sys
import numpy as np
import h5py
import awkward as ak

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def load(path):
    with h5py.File(path, "r") as f:
        grp = f["Events"]
        arr = ak.from_buffers(
            ak.forms.from_json(grp.attrs["form"]),
            grp.attrs["length"],
            {k: np.asarray(v) for k, v in grp.items()},
        )
    return arr


def column(arr, name):
    """Fully flatten one gamma_subshowers column across events + interactions."""
    return arr["gamma_subshowers"][name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", help="Path to the SIREN HDF5 output file")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    args = parser.parse_args()

    print(f"\nLoading {args.hdf5} ...")
    arr = load(args.hdf5)
    if "gamma_subshowers" not in arr.fields:
        print("  no 'gamma_subshowers' field in this file -- was it written with "
              "develop_blobs=True?")
        sys.exit(1)

    # per-interaction subshower counts, and flat columns over ALL subshowers
    n_per_int = ak.to_numpy(ak.flatten(ak.num(column(arr, "subshower_alpha"), axis=2)))
    alpha  = ak.to_numpy(ak.flatten(column(arr, "subshower_alpha"),     axis=None)).astype(float)
    beta   = ak.to_numpy(ak.flatten(column(arr, "subshower_beta"),      axis=None)).astype(float)
    amp    = ak.to_numpy(ak.flatten(column(arr, "subshower_amplitude"), axis=None)).astype(float)
    # parent: real output has subshower_parent_pid/_inst (ints); the mock has a
    # subshower_parent string tag. Build a uniform string key either way.
    gs_fields = arr["gamma_subshowers"].fields
    if "subshower_parent_pid" in gs_fields:
        pid_flat = ak.to_list(ak.flatten(column(arr, "subshower_parent_pid"), axis=None))
        parent = [str(int(p)) for p in pid_flat]
    else:
        parent = ak.to_list(ak.flatten(column(arr, "subshower_parent"), axis=None))

    n_events = len(arr)
    n_int    = len(n_per_int)
    n_sub    = len(alpha)
    print(f"Events: {n_events}   interactions: {n_int}   gamma subshowers: {n_sub}\n")

    if n_sub == 0:
        print("  no gamma subshowers were produced -- check the model/pythia store "
              "paths and that interactions carry a Hadrons blob.")
        sys.exit(1)

    # -- 1. NaN / Inf --------------------------------------------------------
    print("-- NaN / Inf -----------------------------------------------------------")
    ok = True
    for name, a in [("subshower_alpha", alpha), ("subshower_beta", beta),
                    ("subshower_amplitude", amp)]:
        n_nan, n_inf = int(np.sum(np.isnan(a))), int(np.sum(np.isinf(a)))
        ok &= check(f"{name}: no NaN/Inf", (n_nan == 0 and n_inf == 0),
                    f"{n_nan} NaN, {n_inf} Inf")
    if not ok:
        print("\nCritical NaN/Inf found -- fix before proceeding.\n")
        sys.exit(1)

    # -- 2. Positivity / structure ------------------------------------------
    print("\n-- Positivity & structure ---------------------------------------------")
    check("subshower_alpha > 0",     np.all(alpha > 0), f"min={alpha.min():.3g}")
    check("subshower_beta  > 0",     np.all(beta  > 0), f"min={beta.min():.3g}")
    check("subshower_amplitude > 0", np.all(amp   > 0), f"min={amp.min():.3g}")
    check("every parent tag is a non-empty string",
          all(isinstance(p, str) and len(p) > 0 for p in parent))
    check(">=1 subshower per interaction",
          np.all(n_per_int >= 1),
          f"{int(np.sum(n_per_int < 1))} interactions with 0")

    # -- 3. Summary stats (informational) -----------------------------------
    print("\n-- Summary -------------------------------------------------------------")
    print(f"  subshowers / interaction : mean {n_per_int.mean():.2f}  "
          f"min {n_per_int.min()}  max {n_per_int.max()}")
    # species = parent tag with the "(k)" instance index stripped
    species = [p.split("(")[0] for p in parent]
    uniq, counts = np.unique(species, return_counts=True)
    order = np.argsort(-counts)
    print("  subshowers by parent species:")
    for i in order:
        print(f"      {uniq[i]:6s} {counts[i]:8d}  ({100*counts[i]/n_sub:5.2f}%)")
    # TRUE per-particle m-distribution: group subshowers by parent WITHIN each
    # interaction. Parent tags (e.g. "pi0(1)") are only unique within one
    # interaction -- they repeat across events -- so we must NOT flatten them
    # globally here, or every recurring tag looks like an m>=2 particle.
    import collections
    if "subshower_parent_pid" in gs_fields:
        pid_nested  = ak.to_list(column(arr, "subshower_parent_pid"))
        inst_nested = ak.to_list(column(arr, "subshower_parent_inst"))
        parent_nested = [[list(zip(p, i)) for p, i in zip(ev_p, ev_i)]
                         for ev_p, ev_i in zip(pid_nested, inst_nested)]
    else:
        parent_nested = ak.to_list(column(arr, "subshower_parent"))
    m_counter = collections.Counter()
    for _ev in parent_nested:
        for _interaction in _ev:
            for _tag, _m in collections.Counter(
                    [tuple(t) if isinstance(t, list) else t for t in _interaction]).items():
                m_counter[_m] += 1
    n_parents = sum(m_counter.values())
    print(f"  parent particles: {n_parents}")
    for _m in sorted(m_counter):
        print(f"      m={_m}: {m_counter[_m]:7d}  ({100*m_counter[_m]/max(n_parents,1):.2f}%)")
    m_ge2 = sum(v for mm, v in m_counter.items() if mm >= 2)
    print(f"      m>=2 fraction: {100*m_ge2/max(n_parents,1):.2f}%  "
          f"(expect a few % after the L_char veto)")
    print(f"  alpha  : [{alpha.min():.2f}, {alpha.max():.2f}]  median {np.median(alpha):.2f}")
    print(f"  beta   : [{beta.min():.4f}, {beta.max():.4f}]  median {np.median(beta):.4f}")
    print(f"  amplitude: [{amp.min():.3g}, {amp.max():.3g}]  median {np.median(amp):.3g}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0, 0].hist(alpha, bins=50); ax[0, 0].set_title("alpha"); ax[0, 0].set_yscale("log")
        ax[0, 1].hist(beta, bins=50);  ax[0, 1].set_title("beta");  ax[0, 1].set_yscale("log")
        ax[1, 0].hist(np.log10(amp), bins=50); ax[1, 0].set_title("log10(amplitude)")
        ax[1, 1].hist(n_per_int, bins=range(0, n_per_int.max() + 2))
        ax[1, 1].set_title("subshowers / interaction")
        fig.tight_layout()
        out = args.hdf5.rsplit(".", 1)[0] + "_gamma_subshowers.png"
        fig.savefig(out, dpi=150)
        print(f"\n  wrote {out}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
