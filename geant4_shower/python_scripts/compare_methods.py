"""
compare_methods.py
==================
Summary comparison of THREE profile-generation methods against the Geant4
truth, as a function of shower energy.  Each metric is boiled down to a single
number per (method, energy) so the comparison collapses into clean curves.

Methods (a ladder of increasing realism):

  Fixed Single Gamma    one gamma fit to the G4 MEAN profile at each energy,
                        deterministic -> every shower identical, zero
                        fluctuation.  The "averaged blob" baseline.
  Sampled Single Gamma  one gamma, (alpha, beta) drawn from the learned m=1
                        distribution + a log-normal yield draw.
  Gamma Mixture Model   the full sampler (sum of m gammas).

Metrics per (method, energy):
  rms_yield      std of total-yield distribution            -> fluctuation SIZE
  relrms_yield   std/mean of total yield  (sigma/mu)         -> relative fluctuation
  resid_yield    (mean_model - mean_G4)/mean_G4              -> bias in average light
  resid_dmax     mean(dmax_model) - mean(dmax_G4)  [cm]      -> bias in peak depth

Yield recalibration  (the "correction")
----------------------------------------
The raw sampler sets the total light from a log-normal N draw whose spread
(yield_logsigma) is estimated from the FITTED amplitude sums (sum of A_i of the
gamma mixture).  Those amplitude sums carry extra scatter from the fit itself
(degeneracy between components, BIC over-splitting), so their spread exceeds
the true shower-to-shower spread of the total photon count -> the model
OVER-fluctuates (model sigma/mu sits above G4).

Total light and longitudinal SHAPE are separate physical quantities, so with
--fix-yield (default) each sampled profile's total is recalibrated directly to
G4's own total-photon-count distribution (N_total): normalise the sampled shape
to unit sum, then multiply by a log-normal drawn from the log-mean/log-sigma of
G4 N_total at that energy.  Then the total-yield mean AND sigma/mu match G4 by
construction, while the shape (and thus depth-of-max) is untouched.  This is a
legitimate build-time calibration (the model already derives its yield stats
from these same G4 files) -- just from the cleaner observable.  Pass
--no-fix-yield to see the raw amplitude-based (over-fluctuating) behaviour.

Run
---
    python compare_methods.py --g4-dir ../output \
        --model ../output/shower_model.pkl --species pip

Outputs (into --outdir, default outputs/results), four separate figures + csv:
    yield_fluctuation_rms_<sp>.png
    relative_fluctuation_<sp>.png
    residual_mean_yield_<sp>.png
    residual_shower_max_<sp>.png
    compare_metrics_<sp>.csv
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
#  cosmetics
# ---------------------------------------------------------------------------
G4_LABEL = "Geant4 Truth"
G4_COLOR = "#222222"

# tag -> (descriptive label, colour)
METHOD_META = {
    "M1": ("Fixed Single Gamma",   "#4682B4"),   # steel blue
    "M2": ("Sampled Single Gamma", "#BA55D3"),   # medium orchid
    "M3": ("Gamma Mixture Model",  "#FA8072"),   # salmon
}

SPECIES_LATEX = {
    "pip": r"$\pi^{+}$", "pim": r"$\pi^{-}$", "pi0": r"$\pi^{0}$",
    "Kp": r"$K^{+}$", "Km": r"$K^{-}$",
    "KS": r"$K^{0}_{S}$", "KL": r"$K^{0}_{L}$",
    "p": r"$p$", "n": r"$n$",
}


# ---------------------------------------------------------------------------
#  per-ensemble reductions
# ---------------------------------------------------------------------------
def _yields(P):
    """Total light per shower (bin sum)."""
    return P.sum(axis=1)


def _dmax(P, x):
    """Depth (cm) of the profile maximum, per shower."""
    return x[np.argmax(P, axis=1)]


def ensemble_stats(P, x):
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
    """Fixed Single Gamma: deterministic single gamma fit to the G4 mean profile."""
    fit = fit_profile(x, g4_mean_profile, Kmax=1)
    prof = fit["N"] * fit["w"][0] * _kernel(x, fit["alpha"][0], fit["beta"][0])
    return prof, fit


def method2_sample(interp, pid, E, x, rng):
    """Sampled Single Gamma: (alpha,beta) from the m=1 distribution + yield draw.
    Returns (profile, N) or None if there is no m=1 model at this energy."""
    mc = interp.mean_cov(pid, E, 1)
    if mc is None:
        return None
    mean, cov = mc
    z = rng.multivariate_normal(mean, cov)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = np.clip(z, mean - 2.5 * sd, mean + 2.5 * sd)
    w, alpha, beta = _from_z(z, 1)
    N = interp.yield_mean(pid, E)
    s = interp.yield_logsigma(pid, E)
    if s > 0:
        N *= np.exp(rng.normal(0.0, s))
    return N * w[0] * _kernel(x, alpha[0], beta[0]), N


def _finalize(profs, g4_N, rng, fix_yield):
    """Stack the sampled profiles.  With fix_yield, recalibrate each profile's
    total light to G4's own N_total distribution (log-normal), decoupled from the
    sampled shape: total-yield mean and sigma/mu then match G4 by construction,
    while the shape (depth-of-max, width) is untouched."""
    if not profs:
        return None
    P = np.array(profs)
    if fix_yield:
        logN = np.log(np.maximum(np.asarray(g4_N, float), 1e-30))
        mu, sig = float(logN.mean()), float(logN.std())
        shape = P / P.sum(axis=1, keepdims=True)          # unit-sum shapes
        Y = np.exp(rng.normal(mu, sig, size=len(P)))       # G4-calibrated total
        P = shape * Y[:, None]
    return P


def build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng, fix_yield=True):
    g4 = library[pid][E]["profiles"]
    g4_mean = g4.mean(axis=0)
    g4_N = library[pid][E].get("N_total")
    if g4_N is None:
        g4_N = g4.sum(axis=1)

    # Fixed Single Gamma (deterministic; NOT recalibrated -- it is the baseline)
    p1, _ = method1_profile(x, g4_mean)
    M1 = np.tile(p1, (n_sample, 1))

    # Sampled Single Gamma
    m2p, ok = [], True
    for _ in range(n_sample):
        r = method2_sample(interp, pid, E, x, rng)
        if r is None:
            ok = False
            break
        m2p.append(r[0])
    M2 = _finalize(m2p, g4_N, rng, fix_yield) if ok else None

    # Gamma Mixture Model (full)
    m3p = [sampler.sample_profile(pid, E, rng, x=x)[0] for _ in range(n_sample)]
    M3 = _finalize(m3p, g4_N, rng, fix_yield)

    return dict(G4=g4, M1=M1, M2=M2, M3=M3)


# ---------------------------------------------------------------------------
#  driver for one species
# ---------------------------------------------------------------------------
def run_species(interp, sampler, library, pid, n_sample, seed, outdir, fix_yield=True):
    name = PID_TO_NAME.get(pid, str(pid))
    energies = sorted(library[pid].keys())
    rng = np.random.default_rng(seed)

    rows = []
    for E in energies:
        x = library[pid][E]["z_centers"]
        ens = build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng,
                              fix_yield=fix_yield)
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
    _make_all_plots(rows, name, outdir)
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


def _one_plot(rows, key, title, ylabel, outpath, method_tags,
              include_g4=False, logy=False, hline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    if hline is not None:
        ax.axhline(hline, color="#999999", ls="--", lw=1.3, zorder=1)

    def _draw(tag, label, color, dashed=False):
        E, v = _series(rows, tag, key)
        if len(E) == 0:
            return
        ax.plot(E, v, ls="--" if dashed else "-", color=color, lw=2.6,
                marker="o", ms=10, markeredgecolor="white", markeredgewidth=1.4,
                alpha=0.85, label=label, zorder=5 if dashed else 4)

    if include_g4:
        _draw("G4", G4_LABEL, G4_COLOR, dashed=True)
    for tag in method_tags:
        label, color = METHOD_META[tag]
        _draw(tag, label, color)

    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Shower Energy [GeV]", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.tick_params(axis="both", which="major", labelsize=13, length=6)
    ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
    ax.legend(fontsize=13, framealpha=0.92, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  wrote {outpath}")


def _make_all_plots(rows, name, outdir):
    sp = SPECIES_LATEX.get(name, name)

    # 1) absolute RMS  (drop Fixed Single Gamma: it is exactly 0 -> invalid on log)
    _one_plot(
        rows, "rms_yield",
        f"Cherenkov Photon Yield Fluctuation ({sp})",
        "RMS of Total Yield  [photons]",
        os.path.join(outdir, f"yield_fluctuation_rms_{name}.png"),
        method_tags=["M2", "M3"], include_g4=True, logy=True)

    # 2) relative RMS  (keep Fixed Single Gamma: its flat 0 shows "no fluctuation")
    _one_plot(
        rows, "relrms_yield",
        f"Relative Fluctuation in Cherenkov Photon Yield ({sp})",
        r"$\sigma / \mu$ of Total Yield",
        os.path.join(outdir, f"relative_fluctuation_{name}.png"),
        method_tags=["M1", "M2", "M3"], include_g4=True, logy=False)

    # 3) residual mean yield  (G4 is the zero line)
    _one_plot(
        rows, "resid_yield",
        f"Residual of Mean Yield ({sp})",
        r"$(\mu_{\mathrm{model}} - \mu_{\mathrm{G4}}) \,/\, \mu_{\mathrm{G4}}$",
        os.path.join(outdir, f"residual_mean_yield_{name}.png"),
        method_tags=["M1", "M2", "M3"], include_g4=False, hline=0.0)

    # 4) residual shower max
    _one_plot(
        rows, "resid_dmax",
        f"Residual of Maximum Shower Depth ({sp})",
        r"Mean $d_{\mathrm{max}}$:  Model $-$ Geant4  [cm]",
        os.path.join(outdir, f"residual_shower_max_{name}.png"),
        method_tags=["M1", "M2", "M3"], include_g4=False, hline=0.0)


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
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/result")
    ap.add_argument("--no-fix-yield", dest="fix_yield", action="store_false",
                    help="disable yield/shape decoupling (show the over-fluctuation)")
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
        run_species(interp, sampler, library, pid, args.n_sample, args.seed,
                    args.outdir, fix_yield=args.fix_yield)


if __name__ == "__main__":
    main()
