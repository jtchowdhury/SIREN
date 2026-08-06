"""
compare_fixed2gamma.py
======================
Variant of compare_methods.py where the middle method is a FIXED 2-GAMMA
sampler (exactly m=2, motivated by the sub-cascade multiplicity: median = 2
for pi+ across energies) instead of the single gamma.

Methods compared vs Geant4:
  Current Analytic Approx.   SIREN placeholder (deterministic)
  Fixed 2-Gamma              exactly m=2, params drawn from the learned m=2
                             distribution + G4-calibrated yield
  Gamma Mixture Model        the full BIC sampler (m in {1,2,3})

Rationale: 2 components can capture the asymmetry/two-bump structure a single
gamma cannot, while fixing m=2 avoids the over-splitting (stray m=3 components)
that inflates the full mixture's shape fluctuation at high energy.

All the heavy lifting (yield recalibration, shape/L2 metrics, plotting, fit
residuals) is reused from compare_methods.py -- only the middle sampler and the
labels change.

Run
---
    python compare_fixed2gamma.py --species pip
    # data:  ../output/data ,  model: ../output/data/shower_model.pkl
    # plots: output/plots/alt_result

Outputs mirror compare_methods.py (7 figures + 2 csv), into --outdir.
"""

import os
import argparse
import numpy as np
from scipy.optimize import curve_fit

# reuse everything possible from the main comparison + the model internals
import compare_methods as C
from shower_gamma_model import (
    load_model, load_g4_library, fit_profile, _kernel, _from_z,
    ShowerSampler, NAME_TO_PID, PID_TO_NAME,
    _initial_guess, _mixture, _sort_components,
)

# ---------------------------------------------------------------------------
#  labels/colours: replace "Single Gamma" with "Fixed 2-Gamma" (same colours)
# ---------------------------------------------------------------------------
ALT_META = {
    "M1": ("Current Analytic Approx.", "#4682B4"),      # steel blue
    "SG": ("Sampled Single Gamma",     "#BA55D3"),      # medium orchid
    "F2": ("Sampled 2-Gamma",          "lightseagreen"),
    "M3": ("Gamma Mixture Model",      "#FA8072"),      # salmon
}
FIT_META_ALT = {
    "analytic": ("Analytic (fixed shape)", "#4682B4"),
    "single":   ("Single Gamma Fit",       "#BA55D3"),
    "twogamma": ("2-Gamma Fit",            "lightseagreen"),
    "mixture":  ("Gamma Mixture Fit",       "#FA8072"),
}


# ---------------------------------------------------------------------------
#  fixed-2-gamma sampler (mirror of method2_sample but m=2)
# ---------------------------------------------------------------------------
def method_2gamma_sample(interp, pid, E, x, rng):
    """Exactly two gammas: (w, alpha, beta) drawn from the learned m=2
    distribution + a log-normal yield.  Returns (profile, N) or None if the
    model has no m=2 distribution at this energy."""
    mc = interp.mean_cov(pid, E, 2)
    if mc is None:
        return None
    mean, cov = mc
    z = rng.multivariate_normal(mean, cov)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    z = np.clip(z, mean - 2.5 * sd, mean + 2.5 * sd)
    w, alpha, beta = _from_z(z, 2)
    N = interp.yield_mean(pid, E)
    s = interp.yield_logsigma(pid, E)
    if s > 0:
        N *= np.exp(rng.normal(0.0, s))
    prof = N * sum(w[i] * _kernel(x, alpha[i], beta[i]) for i in range(2))
    return prof, N


def build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng, fix_yield=True):
    g4 = library[pid][E]["profiles"]
    g4_mean = g4.mean(axis=0)
    g4_N = library[pid][E].get("N_total")
    if g4_N is None:
        g4_N = g4.sum(axis=1)

    p1, _ = C.method1_profile(x, g4_mean, E)
    M1 = np.tile(p1, (n_sample, 1))

    # Sampled Single Gamma (m=1)
    sgp, ok_sg = [], True
    for _ in range(n_sample):
        r = C.method2_sample(interp, pid, E, x, rng)
        if r is None:
            ok_sg = False
            break
        sgp.append(r[0])
    SG = C._finalize(sgp, g4_N, rng, fix_yield) if ok_sg else None

    # Fixed 2-Gamma (m=2)
    f2p, ok2 = [], True
    for _ in range(n_sample):
        r = method_2gamma_sample(interp, pid, E, x, rng)
        if r is None:
            ok2 = False
            break
        f2p.append(r[0])
    F2 = C._finalize(f2p, g4_N, rng, fix_yield) if ok2 else None

    m3p = [sampler.sample_profile(pid, E, rng, x=x)[0] for _ in range(n_sample)]
    M3 = C._finalize(m3p, g4_N, rng, fix_yield)

    return dict(G4=g4, M1=M1, SG=SG, F2=F2, M3=M3)


# ---------------------------------------------------------------------------
#  fit fidelity: analytic / EXACT-2-gamma fit / m-gamma(BIC) fit  vs G4
# ---------------------------------------------------------------------------
def fit_exact2(x, y):
    """Least-squares fit of EXACTLY two gammas (no BIC selection)."""
    p0, bounds = _initial_guess(x, y, 2)
    popt, _ = curve_fit(_mixture, x, y, p0=p0, bounds=bounds, maxfev=6000)
    A = np.array(popt[0::3]); al = np.array(popt[1::3]); be = np.array(popt[2::3])
    A, al, be = _sort_components(A, al, be)
    Asum = A.sum()
    return dict(m=2, w=(A / Asum if Asum > 0 else np.ones(2) / 2),
                alpha=al, beta=be, N=(Asum if Asum > 0 else float(np.trapz(y, x))))


def fit_l2_forms_alt(g4_profiles, x, E, n_fit, rng):
    n = len(g4_profiles)
    idx = rng.choice(n, size=min(n_fit, n), replace=False)
    alpha = 0.3 + 0.7 * np.log(E / C.HAD_EC_GEV)
    ashape = _kernel(np.asarray(x, float) / C.X0_ICE_CM, alpha, 0.9)
    aa = float(np.dot(ashape, ashape))
    res = {"analytic": [], "single": [], "twogamma": [], "mixture": []}
    for i in idx:
        g = g4_profiles[i]
        denom = float(np.dot(g, g))
        if denom <= 0:
            continue
        c = float(np.dot(g, ashape)) / aa if aa > 0 else 0.0
        res["analytic"].append(float(np.sum((g - c * ashape) ** 2)) / denom)
        f1 = fit_profile(x, g, Kmax=1)
        res["single"].append(float(np.sum((g - C._fit_curve(x, f1)) ** 2)) / denom)
        try:
            f2 = fit_exact2(x, g)
            res["twogamma"].append(float(np.sum((g - C._fit_curve(x, f2)) ** 2)) / denom)
        except Exception:
            pass
        fm = fit_profile(x, g, Kmax=3)
        res["mixture"].append(float(np.sum((g - C._fit_curve(x, fm)) ** 2)) / denom)
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in res.items()}


# ---------------------------------------------------------------------------
#  driver + plots (mirror compare_methods, with ALT labels)
# ---------------------------------------------------------------------------
def run_species(interp, sampler, library, pid, n_sample, seed, outdir,
                fix_yield=True, n_fit=150):
    name = PID_TO_NAME.get(pid, str(pid))
    energies = sorted(library[pid].keys())
    rng = np.random.default_rng(seed)

    rows, fit_rows = [], []
    for E in energies:
        x = library[pid][E]["z_centers"]
        ens = build_ensembles(interp, sampler, library, pid, E, x, n_sample, rng,
                              fix_yield=fix_yield)
        g4 = C.ensemble_stats(ens["G4"], x)
        g4_ms, g4_ss = C._shape_profiles(ens["G4"])
        for tag in ("G4", "M1", "SG", "F2", "M3"):
            P = ens[tag]
            if P is None:
                continue
            s = C.ensemble_stats(P, x)
            ms, ss = C._shape_profiles(P)
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
        fl = fit_l2_forms_alt(ens["G4"], x, E, n_fit, rng)
        for form, val in fl.items():
            fit_rows.append(dict(species=name, E=E, form=form, fit_l2=val))

    C._write_csv(rows, os.path.join(outdir, f"compare_metrics_{name}.csv"))
    C._write_fit_csv(fit_rows, os.path.join(outdir, f"compare_fitL2_{name}.csv"))
    _make_alt_plots(rows, fit_rows, name, outdir)
    return rows, fit_rows


def _make_alt_plots(rows, fit_rows, name, outdir):
    sp = C.SPECIES_LATEX.get(name, name)
    p = lambda f: os.path.join(outdir, f)
    op = C._one_plot

    op(rows, "rms_yield",
       f"Cherenkov Photon Yield Fluctuation ({sp})",
       "RMS of Total Yield  [photons]",
       p(f"yield_fluctuation_rms_{name}.png"),
       method_tags=["SG", "F2", "M3"], meta=ALT_META, include_g4=True, logy=True)

    op(rows, "relrms_yield",
       f"Relative Fluctuation in Cherenkov Photon Yield ({sp})",
       r"$\sigma / \mu$ of Total Yield",
       p(f"relative_fluctuation_{name}.png"),
       method_tags=["M1", "SG", "F2", "M3"], meta=ALT_META, include_g4=True, logy=False)

    op(rows, "resid_yield",
       f"Residual of Mean Yield ({sp})",
       r"$(\mu_{\mathrm{model}} - \mu_{\mathrm{G4}}) \,/\, \mu_{\mathrm{G4}}$",
       p(f"residual_mean_yield_{name}.png"),
       method_tags=["M1", "SG", "F2", "M3"], meta=ALT_META, include_g4=False, hline=0.0)

    op(rows, "resid_dmax",
       f"Residual of Maximum Shower Depth ({sp})",
       r"Mean $d_{\mathrm{max}}$:  Model $-$ Geant4  [cm]",
       p(f"residual_shower_max_{name}.png"),
       method_tags=["M1", "SG", "F2", "M3"], meta=ALT_META, include_g4=False, hline=0.0)

    op(rows, "l2_mean",
       f"Mean-Shape $L_2$ vs Geant4 ({sp})",
       r"$\sum_x (\bar S_{\mathrm{G4}} - \bar S_{\mathrm{model}})^2$",
       p(f"l2_mean_shape_{name}.png"),
       method_tags=["M1", "SG", "F2", "M3"], meta=ALT_META, include_g4=False, logy=True)

    op(rows, "l2_fluc",
       f"Shape-Fluctuation $L_2$ vs Geant4 ({sp})",
       r"$\sum_x (\sigma^{S}_{\mathrm{G4}} - \sigma^{S}_{\mathrm{model}})^2$",
       p(f"l2_fluctuation_shape_{name}.png"),
       method_tags=["M1", "SG", "F2", "M3"], meta=ALT_META, include_g4=False, logy=True)

    op(fit_rows, "fit_l2",
       f"Per-Event Reconstruction $L_2$ vs Geant4 ({sp})",
       r"$\langle\, \sum_x (\mathrm{data}-\mathrm{model})^2 / \sum_x \mathrm{data}^2 \,\rangle$  per event",
       p(f"fit_residual_l2_{name}.png"),
       method_tags=["analytic", "single", "twogamma", "mixture"], meta=FIT_META_ALT,
       match_key="form", include_g4=False, logy=True)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output/data")
    ap.add_argument("--model", default=None,
                    help="prebuilt model pkl (default: <g4-dir>/shower_model.pkl)")
    ap.add_argument("--species", default="pip")
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--n-fit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/alt_result")
    ap.add_argument("--no-fix-yield", dest="fix_yield", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_path = args.model or os.path.join(args.g4_dir, "shower_model.pkl")

    interp = load_model(model_path)
    sampler = ShowerSampler(interp)
    library = load_g4_library(args.g4_dir)

    model_species = sorted(PID_TO_NAME.get(p, str(p)) for p in interp.pid_models)
    print(f"species in model: {model_species}")

    for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
        pid = NAME_TO_PID.get(sp)
        if pid is None:
            print(f"skip unknown species '{sp}'"); continue
        if pid not in library:
            print(f"skip '{sp}': no G4 files in {args.g4_dir}"); continue
        if pid not in interp.pid_models:
            print(f"skip '{sp}': not in the model"); continue
        print(f"== {sp} ==")
        run_species(interp, sampler, library, pid, args.n_sample, args.seed,
                    args.outdir, fix_yield=args.fix_yield, n_fit=args.n_fit)


if __name__ == "__main__":
    main()
