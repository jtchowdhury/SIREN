"""
compare_methods.py
==================
Summary comparison of THREE profile-generation methods against the Geant4
truth, as a function of shower energy.  Each metric is boiled down to a single
number per (method, energy) so the comparison collapses into clean curves.

Methods (a ladder of increasing realism):

  Current Analytic      SIREN's placeholder gamma profile (DISFromSpline):
  Approx.               alpha = 0.3 + 0.7 ln(E/E_c), beta = 0.9, E_c = 0.2 GeV,
                        gamma in radiation lengths (X0 = 39.15 cm).  Deterministic
                        -> zero fluctuation.  Normalised to G4 mean total light.
  Sampled Single Gamma  one gamma, (alpha, beta) drawn from the learned m=1
                        distribution + a log-normal yield draw.
  Gamma Mixture Model   the full sampler (sum of m gammas).

Metrics per (method, energy):
  rms_yield      std of total-yield distribution            -> fluctuation SIZE
  relrms_yield   std/mean of total yield  (sigma/mu)         -> relative fluctuation
  resid_yield    (mean_model - mean_G4)/mean_G4              -> bias in average light
  resid_dmax     mean(dmax_model) - mean(dmax_G4)  [cm]      -> bias in peak depth
  l2_mean        sum_x (meanShape_G4 - meanShape_model)^2    -> MEAN shape match
  l2_fluc        sum_x (stdShape_G4  - stdShape_model)^2     -> shape FLUCTUATION match
     (shapes are unit-area normalised, so these are pure-shape and yield-independent)

Plus a separate "fit fidelity" figure (Andy's L2): for each G4 shower, how well
does each functional FORM represent it -- analytic (fixed shape), single-gamma
fit, m-gamma fit -- as a relative residual sum_x (g-fit)^2 / sum_x g^2, averaged.
This measures the fitting step, not the sampler (a sampled shower has no G4
counterpart to difference against).

Yield recalibration  (--fix-yield, default on)
-----------------------------------------------
The raw sampler's yield spread comes from FITTED amplitude sums, which carry
extra fit scatter and make the model over-fluctuate.  With --fix-yield each
sampled profile's total is recalibrated to G4's own N_total distribution
(unit-normalise the shape, multiply by a log-normal from G4 N_total), so total
yield mean and sigma/mu match G4 by construction while the shape is untouched.

Run
---
    python compare_methods.py --g4-dir ../output/data \
        --model ../output/data/shower_model.pkl --species pip

Outputs (into --outdir), one figure per metric + csv:
    yield_fluctuation_rms_<sp>.png     relative_fluctuation_<sp>.png
    residual_mean_yield_<sp>.png       residual_shower_max_<sp>.png
    l2_mean_shape_<sp>.png             l2_fluctuation_shape_<sp>.png
    fit_residual_l2_<sp>.png
    compare_metrics_<sp>.csv           compare_fitL2_<sp>.csv
"""

import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

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

# tag -> (descriptive label, colour)  [sampling methods]
METHOD_META = {
    "M1": ("Current Analytic Approx.", "#4682B4"),   # steel blue
    "M2": ("Sampled Single Gamma",     "#BA55D3"),   # medium orchid
    "M3": ("Gamma Mixture Model",      "#FA8072"),   # salmon
}

# form -> (label, colour)  [fit-fidelity plot; same colours as the method ladder]
FIT_META = {
    "analytic": ("Analytic (fixed shape)", "#4682B4"),
    "single":   ("Single Gamma Fit",       "#BA55D3"),
    "mixture":  ("Gamma Mixture Fit",       "#FA8072"),
}

SPECIES_LATEX = {
    "pip": r"$\pi^{+}$", "pim": r"$\pi^{-}$", "pi0": r"$\pi^{0}$",
    "Kp": r"$K^{+}$", "Km": r"$K^{-}$",
    "KS": r"$K^{0}_{S}$", "KL": r"$K^{0}_{L}$",
    "p": r"$p$", "n": r"$n$",
}

# SIREN DISFromSpline placeholder constants (hadronic shower in ice)
X0_ICE_CM = 36.08 / 0.9216      # radiation length in ice ~39.15 cm
HAD_EC_GEV = 0.2                # hadronic critical energy [GeV]

# detector resolution: IceCube DOM timing -> Cherenkov emission-depth blur.
# depth floor = v * (time resolution), with v = c/n (light in ice) ~ 45 cm.
C_CM_PER_NS = 29.9792458
N_ICE = 1.33
DOM_TIME_RES_NS = 2.0
DEPTH_RES_CM = C_CM_PER_NS / N_ICE * DOM_TIME_RES_NS      # ~45 cm

# which figures to produce (default: only the per-event L2 + KS panel)
MAKE_PLOTS = {
    "rms_yield":       False,
    "relrms_yield":    False,
    "resid_yield":     False,
    "resid_dmax":      False,
    "l2_mean":         False,
    "l2_fluc":         False,
    "perevent_l2_ks":  True,
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


def _shape_profiles(P):
    """Unit-area-normalise every shower, then return (mean shape, std shape)
    profiles across the ensemble.  Pure shape -> independent of total yield."""
    s = P.sum(axis=1, keepdims=True)
    Ph = P / np.where(s > 0, s, 1.0)
    return Ph.mean(axis=0), Ph.std(axis=0)


# ---------------------------------------------------------------------------
#  the three samplers
# ---------------------------------------------------------------------------
def method1_profile(x, g4_mean_profile, E):
    """Current Analytic Approximation: SIREN's placeholder hadronic longitudinal
    profile (DISFromSpline) -- a gamma in radiation lengths with
        alpha = 0.3 + 0.7*ln(E/E_c),  beta = 0.9,  E_c = 0.2 GeV,  X0 = 39.15 cm.
    Deterministic (zero fluctuation).  Normalised to the G4 mean total light."""
    alpha = 0.3 + 0.7 * np.log(E / HAD_EC_GEV)
    beta = 0.9
    t = np.asarray(x, float) / X0_ICE_CM          # depth [cm] -> radiation lengths
    shape = _kernel(t, alpha, beta)               # unit-area gamma in t
    s = shape.sum()
    if s > 0:
        shape = shape / s * float(g4_mean_profile.sum())
    return shape, dict(alpha=alpha, beta=beta)


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
    """Stack; with fix_yield recalibrate each profile's total to G4's N_total
    distribution (decoupled from shape)."""
    if not profs:
        return None
    P = np.array(profs)
    if fix_yield:
        logN = np.log(np.maximum(np.asarray(g4_N, float), 1e-30))
        mu, sig = float(logN.mean()), float(logN.std())
        shape = P / P.sum(axis=1, keepdims=True)
        Y = np.exp(rng.normal(mu, sig, size=len(P)))
        P = shape * Y[:, None]
    return P


def build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng, fix_yield=True):
    g4 = library[pid][E]["profiles"]
    g4_mean = g4.mean(axis=0)
    g4_N = library[pid][E].get("N_total")
    if g4_N is None:
        g4_N = g4.sum(axis=1)

    # Current Analytic Approx. (deterministic; NOT recalibrated -- it is the baseline)
    p1, _ = method1_profile(x, g4_mean, E)
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
#  fit-fidelity (Andy's L2): how well does each FORM represent a G4 shower
# ---------------------------------------------------------------------------
def _fit_curve(x, fit):
    """Rebuild the fitted profile N * sum_i w_i Gamma(x; alpha_i, beta_i)."""
    return fit["N"] * sum(fit["w"][i] * _kernel(x, fit["alpha"][i], fit["beta"][i])
                          for i in range(fit["m"]))


def fit_l2_forms(g4_profiles, x, E, n_fit, rng):
    """Relative L2 residual  sum(g-fit)^2 / sum(g^2)  of each functional form to
    individual G4 showers, averaged over a subsample.  Forms:
      analytic  fixed SIREN shape, best amplitude scale only
      single    one gamma, least-squares
      mixture   sum of m gammas (BIC), least-squares"""
    n = len(g4_profiles)
    idx = rng.choice(n, size=min(n_fit, n), replace=False)
    alpha = 0.3 + 0.7 * np.log(E / HAD_EC_GEV)
    ashape = _kernel(np.asarray(x, float) / X0_ICE_CM, alpha, 0.9)  # unit-area analytic
    aa = float(np.dot(ashape, ashape))
    res = {"analytic": [], "single": [], "mixture": []}
    for i in idx:
        g = g4_profiles[i]
        denom = float(np.dot(g, g))
        if denom <= 0:
            continue
        c = float(np.dot(g, ashape)) / aa if aa > 0 else 0.0    # best amplitude
        res["analytic"].append(float(np.sum((g - c * ashape) ** 2)) / denom)
        f1 = fit_profile(x, g, Kmax=1)
        res["single"].append(float(np.sum((g - _fit_curve(x, f1)) ** 2)) / denom)
        fm = fit_profile(x, g, Kmax=3)
        res["mixture"].append(float(np.sum((g - _fit_curve(x, fm)) ** 2)) / denom)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in res.items()}


# ---------------------------------------------------------------------------
#  detector-resolution per-event L2 + KS (Andy's criterion)
# ---------------------------------------------------------------------------
def perevent_l2_ks_forms(g4_profiles, x, E, n_fit, rng, sigma_cm):
    """Blur each G4 shower to the detector resolution (Gaussian sigma_cm, v=c/n)
    -- "the best you can do" -- reconstruct that blurred shower with each form,
    and report, per event, the relative L2 and the KS statistic (max gap between
    the two profiles' CDFs, treating each normalized profile as a Cherenkov
    emission-depth distribution).  Only the G4 data is blurred, not the model."""
    n = len(g4_profiles)
    idx = rng.choice(n, size=min(n_fit, n), replace=False)
    binw = float(x[1] - x[0])
    sig_bins = max(sigma_cm / binw, 1e-6)
    alpha = 0.3 + 0.7 * np.log(E / HAD_EC_GEV)
    ashape = _kernel(np.asarray(x, float) / X0_ICE_CM, alpha, 0.9)
    aa = float(np.dot(ashape, ashape))
    out = {f: {"l2": [], "ks": []} for f in ("analytic", "single", "mixture")}
    for i in idx:
        g = gaussian_filter1d(np.asarray(g4_profiles[i], float), sig_bins, mode="constant")
        denom = float(np.dot(g, g))
        if denom <= 0:
            continue
        gpos = np.clip(g, 0.0, None); gsum = gpos.sum()
        cdf_g = np.cumsum(gpos) / gsum if gsum > 0 else None
        c = float(np.dot(g, ashape)) / aa if aa > 0 else 0.0
        fits = {"analytic": c * ashape,
                "single":   _fit_curve(x, fit_profile(x, g, Kmax=1)),
                "mixture":  _fit_curve(x, fit_profile(x, g, Kmax=3))}
        for form, fit in fits.items():
            out[form]["l2"].append(float(np.sum((g - fit) ** 2)) / denom)
            if cdf_g is not None:
                fp = np.clip(fit, 0.0, None); fs = fp.sum()
                if fs > 0:
                    out[form]["ks"].append(
                        float(np.max(np.abs(np.cumsum(fp) / fs - cdf_g))))
    return {f: {"l2": float(np.mean(d["l2"])) if d["l2"] else np.nan,
                "ks": float(np.mean(d["ks"])) if d["ks"] else np.nan}
            for f, d in out.items()}


# ---------------------------------------------------------------------------
#  driver for one species
# ---------------------------------------------------------------------------
def run_species(interp, sampler, library, pid, n_sample, seed, outdir,
                fix_yield=True, n_fit=150, depth_res_cm=DEPTH_RES_CM):
    name = PID_TO_NAME.get(pid, str(pid))
    energies = sorted(library[pid].keys())
    rng = np.random.default_rng(seed)

    need_ens = any(MAKE_PLOTS[k] for k in
                   ("rms_yield", "relrms_yield", "resid_yield",
                    "resid_dmax", "l2_mean", "l2_fluc"))
    need_pe = MAKE_PLOTS["perevent_l2_ks"]

    rows, pe_rows = [], []
    for E in energies:
        x = library[pid][E]["z_centers"]

        if need_ens:
            ens = build_ensembles(interp, sampler, library, pid, E, x, n_sample,
                                  rng, fix_yield=fix_yield)
            g4 = ensemble_stats(ens["G4"], x)
            g4_ms, g4_ss = _shape_profiles(ens["G4"])
            for tag in ("G4", "M1", "M2", "M3"):
                P = ens[tag]
                if P is None:
                    continue
                s = ensemble_stats(P, x)
                ms, ss = _shape_profiles(P)
                rows.append(dict(
                    species=name, E=E, method=tag,
                    mean_yield=s["mean_yield"], rms_yield=s["rms_yield"],
                    relrms_yield=s["relrms_yield"],
                    mean_dmax=s["mean_dmax"], std_dmax=s["std_dmax"],
                    resid_yield=(s["mean_yield"] - g4["mean_yield"]) / g4["mean_yield"],
                    resid_dmax=s["mean_dmax"] - g4["mean_dmax"],
                    l2_mean=float(np.sum((g4_ms - ms) ** 2)),
                    l2_fluc=float(np.sum((g4_ss - ss) ** 2)),
                ))

        if need_pe:
            g4prof = library[pid][E]["profiles"]
            res = perevent_l2_ks_forms(g4prof, x, E, n_fit, rng, depth_res_cm)
            for form, d in res.items():
                pe_rows.append(dict(species=name, E=E, form=form,
                                    l2=d["l2"], ks=d["ks"]))
            print(f"  E={E:8.0f}  " +
                  "  ".join(f"{f}:L2={d['l2']:.2g},KS={d['ks']:.2g}"
                           for f, d in res.items()))

    if need_ens:
        _write_csv(rows, os.path.join(outdir, f"compare_metrics_{name}.csv"))
    if need_pe:
        _write_l2ks_csv(pe_rows, os.path.join(outdir, f"perevent_l2ks_{name}.csv"))
    _make_all_plots(rows, pe_rows, name, outdir, depth_res_cm)
    return rows, pe_rows


# ---------------------------------------------------------------------------
#  outputs
# ---------------------------------------------------------------------------
def _write_csv(rows, path):
    import csv
    cols = ["species", "E", "method", "mean_yield", "rms_yield", "relrms_yield",
            "mean_dmax", "std_dmax", "resid_yield", "resid_dmax", "l2_mean", "l2_fluc"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def _write_fit_csv(fit_rows, path):
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["species", "E", "form", "fit_l2"])
        wr.writeheader()
        for r in fit_rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def _write_l2ks_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["species", "E", "form", "l2", "ks"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {path}")


def _series(rows, key_match, val_key, match_key="method"):
    r = sorted([x for x in rows if x[match_key] == key_match], key=lambda d: d["E"])
    return np.array([d["E"] for d in r]), np.array([d[val_key] for d in r], float)


def _one_plot(rows, key, title, ylabel, outpath, method_tags, meta=METHOD_META,
              match_key="method", include_g4=False, logy=False, hline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    if hline is not None:
        ax.axhline(hline, color="#999999", ls="--", lw=1.3, zorder=1)

    def _draw(tag, label, color, dashed=False):
        E, v = _series(rows, tag, key, match_key=match_key)
        if len(E) == 0:
            return
        ax.plot(E, v, ls="--" if dashed else "-", color=color, lw=2.6,
                marker="o", ms=10, markeredgecolor="white", markeredgewidth=1.4,
                alpha=0.85, label=label, zorder=5 if dashed else 4)

    if include_g4:
        _draw("G4", G4_LABEL, G4_COLOR, dashed=True)
    for tag in method_tags:
        label, color = meta[tag]
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


def _plot_l2_ks(pe_rows, name, outdir, sigma_cm):
    """Per-event reconstruction vs detector-blurred G4: L2 (top) and KS (bottom)
    on a shared energy x-axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = SPECIES_LATEX.get(name, name)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)

    def draw(ax, val_key):
        for form in ("analytic", "single", "mixture"):
            label, color = FIT_META[form]
            E, v = _series(pe_rows, form, val_key, match_key="form")
            if len(E) == 0:
                continue
            ax.plot(E, v, "-", color=color, lw=2.6, marker="o", ms=10,
                    markeredgecolor="white", markeredgewidth=1.4, alpha=0.85,
                    label=label)
        ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=12, length=6)

    draw(ax1, "l2")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"per-event $L_2$   $\sum_x(\mathrm{data}-\mathrm{model})^2/\sum_x\mathrm{data}^2$",
                   fontsize=13)
    ax1.set_title(f"Per-Event Reconstruction vs Detector-Blurred Geant4 ({sp})\n"
                  rf"G4 smeared with Gaussian $\sigma={sigma_cm:.0f}$ cm  ($v=c/n$, 2 ns)",
                  fontsize=15, fontweight="bold", pad=10)
    ax1.legend(fontsize=12, framealpha=0.92, loc="best")

    draw(ax2, "ks")
    ax2.set_xscale("log")
    ax2.set_ylabel("KS statistic  (max CDF gap)", fontsize=14)
    ax2.set_xlabel("Shower Energy [GeV]", fontsize=15)

    fig.tight_layout()
    out = os.path.join(outdir, f"perevent_l2ks_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _make_all_plots(rows, pe_rows, name, outdir, depth_res_cm=DEPTH_RES_CM):
    sp = SPECIES_LATEX.get(name, name)
    p = lambda f: os.path.join(outdir, f)

    if MAKE_PLOTS["rms_yield"]:
        _one_plot(rows, "rms_yield",
                  f"Cherenkov Photon Yield Fluctuation ({sp})",
                  "RMS of Total Yield  [photons]",
                  p(f"yield_fluctuation_rms_{name}.png"),
                  method_tags=["M2", "M3"], include_g4=True, logy=True)

    if MAKE_PLOTS["relrms_yield"]:
        _one_plot(rows, "relrms_yield",
                  f"Relative Fluctuation in Cherenkov Photon Yield ({sp})",
                  r"$\sigma / \mu$ of Total Yield",
                  p(f"relative_fluctuation_{name}.png"),
                  method_tags=["M1", "M2", "M3"], include_g4=True, logy=False)

    if MAKE_PLOTS["resid_yield"]:
        _one_plot(rows, "resid_yield",
                  f"Residual of Mean Yield ({sp})",
                  r"$(\mu_{\mathrm{model}} - \mu_{\mathrm{G4}}) \,/\, \mu_{\mathrm{G4}}$",
                  p(f"residual_mean_yield_{name}.png"),
                  method_tags=["M1", "M2", "M3"], include_g4=False, hline=0.0)

    if MAKE_PLOTS["resid_dmax"]:
        _one_plot(rows, "resid_dmax",
                  f"Residual of Maximum Shower Depth ({sp})",
                  r"Mean $d_{\mathrm{max}}$:  Model $-$ Geant4  [cm]",
                  p(f"residual_shower_max_{name}.png"),
                  method_tags=["M1", "M2", "M3"], include_g4=False, hline=0.0)

    if MAKE_PLOTS["l2_mean"]:
        _one_plot(rows, "l2_mean",
                  f"Mean-Shape $L_2$ vs Geant4 ({sp})",
                  r"$\sum_x (\bar S_{\mathrm{G4}} - \bar S_{\mathrm{model}})^2$",
                  p(f"l2_mean_shape_{name}.png"),
                  method_tags=["M1", "M2", "M3"], include_g4=False, logy=True)

    if MAKE_PLOTS["l2_fluc"]:
        _one_plot(rows, "l2_fluc",
                  f"Shape-Fluctuation $L_2$ vs Geant4 ({sp})",
                  r"$\sum_x (\sigma^{S}_{\mathrm{G4}} - \sigma^{S}_{\mathrm{model}})^2$",
                  p(f"l2_fluctuation_shape_{name}.png"),
                  method_tags=["M1", "M2", "M3"], include_g4=False, logy=True)

    if MAKE_PLOTS["perevent_l2_ks"]:
        _plot_l2_ks(pe_rows, name, outdir, depth_res_cm)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output/data",
                    help="dir with shower_<name>_E<E>GeV.h5 files")
    ap.add_argument("--model", default=None,
                    help="prebuilt model pkl (default: <g4-dir>/shower_model.pkl)")
    ap.add_argument("--species", default="pip",
                    help="comma-separated species names, e.g. pip,pim,p")
    ap.add_argument("--n-sample", type=int, default=2000,
                    help="showers sampled per method per energy")
    ap.add_argument("--n-fit", type=int, default=150,
                    help="G4 showers subsampled for the per-event L2/KS (slow)")
    ap.add_argument("--depth-res-cm", type=float, default=DEPTH_RES_CM,
                    help="detector depth resolution for the G4 blur (default ~45 cm, v=c/n, 2 ns)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/result")
    ap.add_argument("--no-fix-yield", dest="fix_yield", action="store_false",
                    help="disable yield recalibration (show the over-fluctuation)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_path = args.model or os.path.join(args.g4_dir, "shower_model.pkl")

    # The per-event L2/KS plot is pure fitting of G4 -- it needs no model.
    # Only load the sampler model if an ensemble-based plot is enabled.
    need_ens = any(MAKE_PLOTS[k] for k in
                   ("rms_yield", "relrms_yield", "resid_yield",
                    "resid_dmax", "l2_mean", "l2_fluc"))

    library = load_g4_library(args.g4_dir)
    print(f"species in G4 dir ({args.g4_dir}): "
          f"{sorted(PID_TO_NAME.get(p, str(p)) for p in library)}")

    interp = sampler = None
    if need_ens:
        interp = load_model(model_path)
        sampler = ShowerSampler(interp)
        print(f"species in model  ({model_path}): "
              f"{sorted(PID_TO_NAME.get(p, str(p)) for p in interp.pid_models)}")

    for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
        pid = NAME_TO_PID.get(sp)
        if pid is None:
            print(f"skip unknown species '{sp}'"); continue
        if pid not in library:
            print(f"skip '{sp}': no shower_{sp}_E*GeV.h5 in {args.g4_dir}"); continue
        if need_ens and pid not in interp.pid_models:
            print(f"skip '{sp}': not in the model -- rebuild shower_model.pkl "
                  f"after its G4 files exist"); continue
        print(f"== {sp} ==")
        run_species(interp, sampler, library, pid, args.n_sample, args.seed,
                    args.outdir, fix_yield=args.fix_yield, n_fit=args.n_fit,
                    depth_res_cm=args.depth_res_cm)


if __name__ == "__main__":
    main()
