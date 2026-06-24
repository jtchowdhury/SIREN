#!/usr/bin/env python3
"""
check_subshowers.py  —  sanity-check subshower output from DISFromSpline

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

def flat(arr, key):
    return ak.to_numpy(ak.flatten(arr[key])).astype(float)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", help="Path to HDF5 output file")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    args = parser.parse_args()

    print(f"\nLoading {args.hdf5} ...")
    arr = load(args.hdf5)

    N      = flat(arr, "subshower_N")
    E_sub  = flat(arr, "subshower_E_sub")
    alpha  = flat(arr, "subshower_alpha")
    beta   = flat(arr, "subshower_beta")

    # Reconstruct E_had = N * E_sub (stored per-subshower energy times count)
    E_had  = N * E_sub

    n_events = len(N)
    print(f"Events loaded: {n_events}\n")

    # ── 1. NaN / Inf checks ───────────────────────────────────────────────────
    print("── NaN / Inf ──────────────────────────────────────────────────────────")
    all_ok = True
    for name, arr_ in [("subshower_N", N), ("subshower_E_sub", E_sub),
                        ("subshower_alpha", alpha), ("subshower_beta", beta)]:
        n_nan = np.sum(np.isnan(arr_))
        n_inf = np.sum(np.isinf(arr_))
        ok = (n_nan == 0) and (n_inf == 0)
        all_ok &= check(f"{name}: no NaN/Inf",
                        ok, f"{n_nan} NaN, {n_inf} Inf")

    if not all_ok:
        print("\nCritical NaN/Inf found — fix before proceeding.\n")
        sys.exit(1)

    # ── 2. Range checks ───────────────────────────────────────────────────────
    print("\n── Range checks ───────────────────────────────────────────────────────")
    check("subshower_N >= 1 always",
          np.all(N >= 1),
          f"min={N.min():.0f}")

    check("subshower_N is integer-valued",
          np.all(np.abs(N - np.round(N)) < 0.5),
          f"max deviation={np.max(np.abs(N - np.round(N))):.2e}")

    check("subshower_alpha > 0 always",
          np.all(alpha > 0),
          f"min={alpha.min():.3f}")

    check("subshower_beta ~ 0.9 (constant)",
          np.allclose(beta, 0.9, atol=1e-6),
          f"mean={beta.mean():.4f}, std={beta.std():.2e}")

    check("subshower_E_sub <= 1.0 GeV + rounding",
          np.all(E_sub <= 1.05),   # 5% tolerance for rounding
          f"max={E_sub.max():.4f} GeV")

    check("subshower_E_sub > 0",
          np.all(E_sub > 0),
          f"min={E_sub.min():.4f} GeV")

    # ── 3. Internal consistency ───────────────────────────────────────────────
    print("\n── Internal consistency ───────────────────────────────────────────────")

    # alpha should equal 0.3 + 0.7 * ln(E_had / 0.2)
    E_c = 0.2
    alpha_expected = 0.3 + 0.7 * np.log(E_had / E_c)
    alpha_residual = np.abs(alpha - alpha_expected)
    check("alpha == 0.3 + 0.7*ln(E_had/0.2)",
          np.all(alpha_residual < 1e-6),
          f"max residual={alpha_residual.max():.2e}")

    # N should equal round(E_had / 1.0), i.e. round(E_had)
    N_expected = np.maximum(1, np.round(E_had / 1.0))
    check("N == max(1, round(E_had / E_sub_threshold))",
          np.all(np.abs(N - N_expected) < 0.5),
          f"mismatches={(np.abs(N - N_expected) >= 0.5).sum()}")

    # ── 4. Physical plausibility ──────────────────────────────────────────────
    print("\n── Physical plausibility ──────────────────────────────────────────────")
    print(f"  subshower_N   : min={N.min():.0f}  median={np.median(N):.0f}"
          f"  max={N.max():.0f}")
    print(f"  E_had (GeV)   : min={E_had.min():.2f}  median={np.median(E_had):.2f}"
          f"  max={E_had.max():.2f}")
    print(f"  subshower_alpha: min={alpha.min():.3f}  median={np.median(alpha):.3f}"
          f"  max={alpha.max():.3f}")
    print(f"  subshower_beta : {beta.mean():.4f} (constant by construction)")
    print(f"  subshower_E_sub: min={E_sub.min():.4f}  max={E_sub.max():.4f} GeV")

    # Warn if very low alpha (gamma_p_inv becomes numerically tricky near 0)
    n_low_alpha = np.sum(alpha < 1.0)
    if n_low_alpha > 0:
        print(f"{WARN}  {n_low_alpha} events have alpha < 1.0 — "
              f"gamma_p_inv quantiles may be inaccurate at low shape values")

    # ── 5. Optional plots ─────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available — skipping plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Subshower sanity check", fontsize=13)

        axes[0, 0].hist(np.log10(N), bins=60, color="steelblue", edgecolor="none")
        axes[0, 0].set_xlabel("log10(subshower_N)")
        axes[0, 0].set_ylabel("Events")
        axes[0, 0].set_title("Number of sub-showers")

        axes[0, 1].hist(alpha, bins=60, color="darkorange", edgecolor="none")
        axes[0, 1].set_xlabel("subshower_alpha")
        axes[0, 1].set_title("Gamma shape parameter α")

        axes[1, 0].hist(E_sub, bins=60, color="seagreen", edgecolor="none")
        axes[1, 0].set_xlabel("subshower_E_sub (GeV)")
        axes[1, 0].set_title("Energy per sub-shower")

        # alpha vs log10(E_had): should be linear
        sample = np.random.choice(len(E_had), min(5000, len(E_had)), replace=False)
        axes[1, 1].scatter(np.log10(E_had[sample]), alpha[sample],
                           s=1, alpha=0.3, color="purple")
        log_E_range = np.linspace(np.log10(E_had.min()), np.log10(E_had.max()), 100)
        E_range = 10 ** log_E_range
        axes[1, 1].plot(log_E_range,
                        0.3 + 0.7 * np.log(E_range / E_c),
                        "r-", lw=1.5, label="expected")
        axes[1, 1].set_xlabel("log10(E_had / GeV)")
        axes[1, 1].set_ylabel("alpha")
        axes[1, 1].set_title("α vs E_had (should be linear)")
        axes[1, 1].legend(fontsize=8)

        out = args.hdf5.replace(".hdf5", "_subshower_check.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"\n  Plot saved to {out}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()
