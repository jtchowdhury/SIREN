"""
test_offset_gamma.py  (standalone diagnostic -- does NOT modify the model)
==========================================================================
Question: is the "every gamma starts at 0" assumption hurting us?  A sub-cascade
ignites stochastically at some depth z0 downstream, but the current kernel
x^(a-1) e^(-b x) is pinned to the origin and can only fake a delayed start by
inflating alpha (which is exactly the degenerate regime that spoils sampling).

For each G4 profile we fit THREE models and compare residuals:
  (1) single      : one zero-origin gamma          (what the model uses now)
  (2) shift       : one SHIFTED gamma (x-z0)_+      (adds a start position z0)
  (3) two         : two zero-origin gammas          (what BIC keeps choosing)

The decisive comparison is (2) vs (3):
  * if the shifted single gamma matches the two-gamma residual, then the second
    gamma was only compensating for the missing start position -> an explicit
    offset should fix pi0 over-splitting and stabilise the mixture.
  * if the shifted single is no better than the plain single, the second gamma
    is capturing real structure a shift can't -> the offset won't help, lean on
    the resolution/separation gate instead.

Models 1 & 3 reuse shower_gamma_model's own _mixture/_initial_guess so the
comparison is apples-to-apples with the real pipeline.

Run (on the cluster, where the G4 .h5 live):
    python test_offset_gamma.py --species pi0,pip
    # fast: ~100 showers/energy by default; --n to change, --n-jobs -1 = all cores
Outputs a console table + summary plots to output/plots/diagnostics/.
"""

import os
import argparse
import numpy as np
from scipy.optimize import curve_fit

from shower_gamma_model import (
    load_g4_library, _mixture, _initial_guess, NAME_TO_PID, PID_TO_NAME,
)


# ---------------------------------------------------------------------------
#  shifted single gamma:  A * (x - z0)_+^(alpha-1) * exp(-beta (x - z0))
# ---------------------------------------------------------------------------
def _shift_gamma(x, A, z0, alpha, beta):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x > z0
    d = x[m] - z0
    out[m] = A * np.power(d, alpha - 1.0) * np.exp(-beta * d)
    return out


def _shift_guess(x, y):
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    ymax = y.max() if y.max() > 0 else 1.0
    ipk = int(np.argmax(y)); xpk = float(x[ipk])
    above = np.where(y > 0.05 * ymax)[0]           # onset = first 5%-of-max bin
    z0 = float(x[above[0]]) if len(above) else 0.0
    z0 = min(max(z0, 0.0), max(xpk - dx, 0.0))
    a0 = 4.0
    b0 = (a0 - 1.0) / max(xpk - z0, dx)
    k = _shift_gamma(np.array([xpk]), 1.0, z0, a0, b0)[0]
    A0 = ymax / max(k, 1e-12)
    p0 = [A0, z0, a0, b0]
    lo = [0.0, 0.0, 1.0, 1e-5]
    hi = [np.inf, max(xpk, dx), 300.0, 0.2]
    return p0, (lo, hi)


# ---------------------------------------------------------------------------
#  fit all three models to one profile; return relative-L2 residuals + z0
# ---------------------------------------------------------------------------
def _fit_one(payload):
    x, y = payload
    x = np.asarray(x, float); y = np.asarray(y, float)
    denom = float(np.sum(y * y)) or 1.0
    r = dict(single=np.nan, shift=np.nan, two=np.nan, z0=np.nan)

    try:                                            # (1) single zero-origin
        p0, (lo, hi) = _initial_guess(x, y, 1)
        popt, _ = curve_fit(_mixture, x, y, p0=p0, bounds=(lo, hi), maxfev=6000)
        r["single"] = float(np.sum((y - _mixture(x, *popt)) ** 2)) / denom
    except Exception:
        pass

    try:                                            # (3) two zero-origin
        p0, (lo, hi) = _initial_guess(x, y, 2)
        popt, _ = curve_fit(_mixture, x, y, p0=p0, bounds=(lo, hi), maxfev=8000)
        r["two"] = float(np.sum((y - _mixture(x, *popt)) ** 2)) / denom
    except Exception:
        pass

    try:                                            # (2) single shifted
        p0, bnds = _shift_guess(x, y)
        popt, _ = curve_fit(_shift_gamma, x, y, p0=p0, bounds=bnds, maxfev=8000)
        r["shift"] = float(np.sum((y - _shift_gamma(x, *popt)) ** 2)) / denom
        r["z0"] = float(popt[1])
    except Exception:
        pass

    return r


# ---------------------------------------------------------------------------
def run_species(library, pid, n_sample, seed, pool):
    name = PID_TO_NAME.get(pid, str(pid))
    rng = np.random.default_rng(seed)
    rows = []                                       # per-energy aggregates
    per_E_examples = {}                             # E -> (x, y) for overlay plot
    for E in sorted(library[pid].keys()):
        x = np.asarray(library[pid][E]["z_centers"], float)
        profs = library[pid][E]["profiles"]
        valid = [np.asarray(p, float) for p in profs if np.asarray(p).max() > 0]
        if not valid:
            continue
        idx = rng.permutation(len(valid))[:n_sample]
        batch = [valid[i] for i in idx]
        payloads = [(x, y) for y in batch]
        res = pool.map(_fit_one, payloads) if pool is not None else \
              [_fit_one(p) for p in payloads]

        s = np.array([d["single"] for d in res], float)
        sh = np.array([d["shift"] for d in res], float)
        tw = np.array([d["two"] for d in res], float)
        z0 = np.array([d["z0"] for d in res], float)
        med_s, med_sh, med_tw = (np.nanmedian(s), np.nanmedian(sh), np.nanmedian(tw))
        gap = med_s - med_tw                         # how much the 2nd gamma helps
        closed = (med_s - med_sh) / gap if gap > 1e-12 else np.nan
        rows.append(dict(E=E, single=med_s, shift=med_sh, two=med_tw,
                         z0=np.nanmedian(z0), z0_std=np.nanstd(z0), closed=closed))
        # keep the median-single profile at this energy for an overlay plot
        j = int(np.nanargmin(np.abs(s - med_s)))
        per_E_examples[E] = (x, batch[j])
        print(f"  {name:4s} E={E:8.0f}  relL2  single={med_s:.4f} "
              f"shift={med_sh:.4f} two={med_tw:.4f}   z0~{np.nanmedian(z0):.0f}cm "
              f"  shift closes {100*closed:5.1f}% of single->two gap"
              if np.isfinite(closed) else
              f"  {name:4s} E={E:8.0f}  relL2  single={med_s:.4f} "
              f"shift={med_sh:.4f} two={med_tw:.4f}   z0~{np.nanmedian(z0):.0f}cm")
    return name, rows, per_E_examples


def _verdict(name, rows):
    """Summarise at the two highest energies (where pi0 over-splits most)."""
    hi = rows[-2:] if len(rows) >= 2 else rows
    cl = np.nanmean([r["closed"] for r in hi if np.isfinite(r["closed"])])
    z0 = np.nanmean([r["z0"] for r in hi])
    print(f"\n  [{name}] high-E: shift closes ~{100*cl:.0f}% of the single->two "
          f"residual gap; typical z0 ~ {z0:.0f} cm")
    if cl >= 0.7:
        print(f"  => the offset largely explains the 2nd gamma for {name}: "
              f"an explicit start position should help.")
    elif cl <= 0.3:
        print(f"  => the offset does NOT explain the 2nd gamma for {name}: "
              f"lean on the resolution/separation gate instead.")
    else:
        print(f"  => partial ({name}): the offset helps but doesn't fully "
              f"account for the 2nd gamma.")


def _plots(name, rows, examples, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Es = [r["E"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(Es, [r["single"] for r in rows], "o-", color="#BA55D3", label="single (zero-origin)")
    a1.plot(Es, [r["shift"] for r in rows], "s-", color="#2E8B57", label="single SHIFTED")
    a1.plot(Es, [r["two"] for r in rows], "^-", color="#FA8072", label="two (zero-origin)")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Shower Energy [GeV]"); a1.set_ylabel("median relative $L_2$ residual")
    a1.set_title(f"Fit residual vs model ({name})"); a1.legend(fontsize=9)
    a1.grid(True, ls=":", alpha=0.6)

    a2.errorbar(Es, [r["z0"] for r in rows], yerr=[r["z0_std"] for r in rows],
                fmt="o-", color="#2E8B57", capsize=3)
    a2.set_xscale("log"); a2.set_xlabel("Shower Energy [GeV]")
    a2.set_ylabel("fitted start position $z_0$ [cm]")
    a2.set_title(f"Shifted-gamma $z_0$ ({name})"); a2.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    p1 = os.path.join(outdir, f"offset_test_{name}.png")
    fig.savefig(p1, dpi=140); plt.close(fig); print(f"  wrote {p1}")

    # overlay of the three fits on a representative highest-energy profile
    Ehi = Es[-1]; x, y = examples[Ehi]
    try:
        ps, _ = curve_fit(_mixture, x, y, p0=_initial_guess(x, y, 1)[0],
                          bounds=_initial_guess(x, y, 1)[1], maxfev=6000)
        pt, _ = curve_fit(_mixture, x, y, p0=_initial_guess(x, y, 2)[0],
                          bounds=_initial_guess(x, y, 2)[1], maxfev=8000)
        g0, gb = _shift_guess(x, y)
        pf, _ = curve_fit(_shift_gamma, x, y, p0=g0, bounds=gb, maxfev=8000)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.plot(x, y, color="0.5", lw=3, alpha=0.6, label="G4 profile")
        ax.plot(x, _mixture(x, *ps), "--", color="#BA55D3", lw=2, label="single")
        ax.plot(x, _shift_gamma(x, *pf), "-", color="#2E8B57", lw=2,
                label=f"shifted (z0={pf[1]:.0f}cm)")
        ax.plot(x, _mixture(x, *pt), ":", color="#FA8072", lw=2, label="two gammas")
        ax.set_xlabel("depth [cm]"); ax.set_ylabel("Cherenkov photons/bin")
        ax.set_title(f"{name}  E={Ehi:.0f} GeV  (representative shower)")
        ax.legend(fontsize=9); ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        p2 = os.path.join(outdir, f"offset_test_{name}_overlay.png")
        fig.savefig(p2, dpi=140); plt.close(fig); print(f"  wrote {p2}")
    except Exception as e:
        print(f"  (overlay skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output/data")
    ap.add_argument("--species", default="pi0,pip")
    ap.add_argument("--n", type=int, default=100, help="showers per energy (subsample)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1=all cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="../output/plots/diagnostics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    library = load_g4_library(args.g4_dir)

    pool = None
    if args.n_jobs != 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=(None if args.n_jobs < 0 else args.n_jobs))
    try:
        for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
            pid = NAME_TO_PID.get(sp)
            if pid is None or pid not in library:
                print(f"skip '{sp}': not found in {args.g4_dir}"); continue
            print(f"== {sp} ==")
            name, rows, examples = run_species(library, pid, args.n, args.seed, pool)
            if rows:
                _verdict(name, rows)
                _plots(name, rows, examples, args.outdir)
    finally:
        if pool is not None:
            pool.close(); pool.join()


if __name__ == "__main__":
    main()
