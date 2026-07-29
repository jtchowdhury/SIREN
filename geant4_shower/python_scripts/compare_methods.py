"""
compare_methods.py
==================
Summary comparison of THREE profile-generation methods against the Geant4
truth, as a function of shower energy.  This is the "scorecard" plot: every
comparison at a given energy is boiled down to a single number, so the whole
thing collapses into a few curves (metric on y, shower energy on x).

The three methods (a ladder of increasing realism):

  M1  "canonical single gamma, no fluctuation"
        One gamma whose (alpha, beta, N) come from a single-gamma fit to the
        G4 MEAN profile at each energy.  Deterministic -> every shower is
        identical.  This is the current "averaged blob" baseline.

  M2  "single gamma, sampled (alpha, beta) from the spline"
        One gamma, but (alpha, beta) are drawn from the m=1 distribution the
        model learned (interpolated in log-E), plus the log-normal yield
        fluctuation.  Adds event-to-event fluctuation, single component.

  M3  "full model"  (shower_gamma_model.ShowerSampler)
        Sum of m gammas, m and (w, alpha, beta) sampled, plus yield.

Metrics per (method, energy), all computed from an ensemble the same way for
the model methods and for the G4 truth:

  RMS(yield)        std of the total-yield distribution     -> fluctuation SIZE
  relRMS(yield)     std/mean of the yield  (sigma/mu)        -> relative fluctuation
                    == the energy-resolution-floor proxy
  residual yield    (mean_model - mean_G4)/mean_G4           -> bias in average light
  residual dmax     mean(dmax_model) - mean(dmax_G4)  [cm]   -> bias in peak depth

M1 has zero fluctuation by construction, so its RMS and relRMS sit on the
floor -- that is the point (it visibly fails the fluctuation metrics).

Run
---
    python compare_methods.py --g4-dir ../output \
        --model ../output/shower_model.pkl --species pip

Outputs (into --outdir, default outputs/results):
    compare_metrics_<species>.png     4-panel scorecard
    compare_metrics_<species>.csv     the underlying numbers
"""

import os
import argparse
import numpy as np

# reuse the model's own building blocks so nothing is re-implemented / guessed
from shower_gamma_model import (
    load_model, load_g4_library, fit_profile, _kernel, _from_z,
    ShowerSampler, NAME_TO_PID, PID_TO_NAME,
)


# ---------------------------------------------------------------------------
#  per-ensemble reductions
# ---------------------------------------------------------------------------
def _yields(P):
    """Total light per shower (bin sum), matching validate_against_g4's convention."""
    return P.sum(axis=1)


def _dmax(P, x):
    """Depth (cm) of the profile maximum, per shower."""
    return x[np.argmax(P, axis=1)]


def ensemble_stats(P, x):
    """Return the scalar summaries used for the metrics."""
    y = _yields(P)
    dm = _dmax(P, x)
    return dict(mean_yield=float(y.mean()),
                rms_yield=float(y.std()),
                relrms_yield=float(y.std() / y.mean()) if y.mean() > 0 else np.nan,
                mean_dmax=float(dm.mean()),
                std_dmax=float(dm.std()))


# ---------------------------------------------------------------------------
#  the three samplers
# ---------------------------------------------------------------------------
def method1_profile(x, g4_mean_profile):
    """M1: deterministic single gamma fit to the G4 mean profile."""
    fit = fit_profile(x, g4_mean_profile, Kmax=1)
    prof = fit["N"] * fit["w"][0] * _kernel(x, fit["alpha"][0], fit["beta"][0])
    return prof, fit


def method2_sample(interp, pid, E, x, rng):
    """M2: single gamma with (alpha,beta) drawn from the m=1 distribution + yield."""
    mc = interp.mean_cov(pid, E, 1)
    if mc is None:
        return None                      # no m=1 model here -> skip this point
    mean, cov = mc
    z = rng.multivariate_normal(mean, cov)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = np.clip(z, mean - 2.5 * sd, mean + 2.5 * sd)   # same truncation as full sampler
    w, alpha, beta = _from_z(z, 1)
    N = interp.yield_mean(pid, E)
    s = interp.yield_logsigma(pid, E)
    if s > 0:
        N *= np.exp(rng.normal(0.0, s))
    return N * w[0] * _kernel(x, alpha[0], beta[0])


def build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng):
    """Return {method: (n_sample, nbins) profile array} for M1/M2/M3, plus G4."""
    g4 = library[pid][E]["profiles"]
    g4_mean = g4.mean(axis=0)

    # M1 deterministic -> replicate the single profile so every reduction is uniform
    p1, _ = method1_profile(x, g4_mean)
    M1 = np.tile(p1, (n_sample, 1))

    # M2 single-gamma sampled
    m2_rows = []
    for _ in range(n_sample):
        p = method2_sample(interp, pid, E, x, rng)
        if p is None:
            m2_rows = None
            break
        m2_rows.append(p)
    M2 = np.array(m2_rows) if m2_rows is not None else None

    # M3 full model
    M3 = np.array([sampler.sample_profile(pid, E, rng, x=x)[0] for _ in range(n_sample)])

    return dict(G4=g4, M1=M1, M2=M2, M3=M3)


# ---------------------------------------------------------------------------
#  driver for one species
# ---------------------------------------------------------------------------
def run_species(interp, sampler, library, pid, n_sample, seed, outdir):
    name = PID_TO_NAME.get(pid, str(pid))
    energies = sorted(library[pid].keys())
    rng = np.random.default_rng(seed)

    rows = []   # flat table: one row per (E, method)
    for E in energies:
        x = library[pid][E]["z_centers"]
        ens = build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng)
        g4 = ensemble_stats(ens["G4"], x)
        for tag in ("G4", "M1", "M2", "M3"):
            P = ens[tag]
            if P is None:
                continue
            s = ensemble_stats(P, x)
            rows.append(dict(
                species=name, E=E, method=tag,
                mean_yield=s["mean_yield"], rms_yield=s["rms_yield"],
                relrms_yield=s["relrms_yield"],
                mean_dmax=s["mean_dmax"], std_dmax=s["std_dmax"],
                resid_yield=(s["mean_yield"] - g4["mean_yield"]) / g4["mean_yield"],
                resid_dmax=s["mean_dmax"] - g4["mean_dmax"],
            ))

    _write_csv(rows, os.path.join(outdir, f"compare_metrics_{name}.csv"))
    _plot(rows, name, os.path.join(outdir, f"compare_metrics_{name}.png"))
    return rows


# ---------------------------------------------------------------------------
#  outputs
# ---------------------------------------------------------------------------
def _write_csv(rows, path):
    import csv
    cols = ["species", "E", "method", "mean_yield", "rms_yield", "relrms_yield",
            "mean_dmax", "std_dmax", "resid_yield", "resid_dmax"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def _series(rows, method, key):
    r = sorted([x for x in rows if x["method"] == method], key=lambda d: d["E"])
    return np.array([d["E"] for d in r]), np.array([d[key] for d in r], float)


def _plot(rows, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {  # tag: (label, color)
        "G4": ("G4 truth", "k"),
        "M1": ("M1  single gamma, no fluct.", "tab:blue"),
        "M2": ("M2  single gamma, sampled", "tab:orange"),
        "M3": ("M3  full model", "tab:red"),
    }

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) absolute RMS of yield ------------------------------------------------
    for tag, (lab, c) in styles.items():
        E, v = _series(rows, tag, "rms_yield")
        if len(E) == 0:
            continue
        ls = "--" if tag == "G4" else "-"
        ax[0, 0].plot(E, v, ls, marker="o", ms=4, color=c, label=lab)
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("shower energy [GeV]"); ax[0, 0].set_ylabel("RMS of total yield")
    ax[0, 0].set_title("(a) yield fluctuation SIZE  (RMS)")
    ax[0, 0].legend(fontsize=8)

    # (b) relative RMS  sigma/mu ----------------------------------------------
    for tag, (lab, c) in styles.items():
        E, v = _series(rows, tag, "relrms_yield")
        if len(E) == 0:
            continue
        ls = "--" if tag == "G4" else "-"
        ax[0, 1].plot(E, v, ls, marker="o", ms=4, color=c, label=lab)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("shower energy [GeV]"); ax[0, 1].set_ylabel(r"$\sigma/\mu$ of yield")
    ax[0, 1].set_title(r"(b) RELATIVE yield fluctuation  ($\sigma/\mu$)")
    ax[0, 1].legend(fontsize=8)

    # (c) residual yield -------------------------------------------------------
    ax[1, 0].axhline(0.0, color="k", ls="--", lw=1)
    for tag, (lab, c) in styles.items():
        if tag == "G4":
            continue
        E, v = _series(rows, tag, "resid_yield")
        if len(E) == 0:
            continue
        ax[1, 0].plot(E, v, "-", marker="o", ms=4, color=c, label=lab)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("shower energy [GeV]")
    ax[1, 0].set_ylabel(r"$(\mu_{\rm model}-\mu_{\rm G4})/\mu_{\rm G4}$")
    ax[1, 0].set_title("(c) residual MEAN yield  (bias)")
    ax[1, 0].legend(fontsize=8)

    # (d) residual shower max --------------------------------------------------
    ax[1, 1].axhline(0.0, color="k", ls="--", lw=1)
    for tag, (lab, c) in styles.items():
        if tag == "G4":
            continue
        E, v = _series(rows, tag, "resid_dmax")
        if len(E) == 0:
            continue
        ax[1, 1].plot(E, v, "-", marker="o", ms=4, color=c, label=lab)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlabel("shower energy [GeV]")
    ax[1, 1].set_ylabel(r"$\overline{d_{\max}}_{\rm model}-\overline{d_{\max}}_{\rm G4}$ [cm]")
    ax[1, 1].set_title("(d) residual shower max  (bias)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"Method comparison vs G4 — {name}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output",
                    help="dir with shower_<name>_E<E>GeV.h5 files")
    ap.add_argument("--model", default=None,
                    help="prebuilt model pkl (default: <g4-dir>/shower_model.pkl)")
    ap.add_argument("--species", default="pip",
                    help="comma-separated species names, e.g. pip,pim,p")
    ap.add_argument("--n-sample", type=int, default=2000,
                    help="showers sampled per method per energy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="outputs/results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_path = args.model or os.path.join(args.g4_dir, "shower_model.pkl")

    interp = load_model(model_path)
    sampler = ShowerSampler(interp)
    library = load_g4_library(args.g4_dir)

    for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
        pid = NAME_TO_PID.get(sp)
        if pid is None:
            print(f"skip unknown species '{sp}'"); continue
        if pid not in library:
            print(f"skip '{sp}': no G4 files loaded for it"); continue
        print(f"== {sp} ==")
        run_species(interp, sampler, library, pid, args.n_sample, args.seed, args.outdir)


if __name__ == "__main__":
    main()
