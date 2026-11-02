#!/usr/bin/env python3
"""
plot_gamma_dis_reweighting.py -- section E: are GammaDISFromSpline events reweightable?

Two GammaDIS samples, both weighted to the SAME physical spectrum (E^-2):
  test       reweight_gamma.npz   generated with E^-1  (the "wrong" spectrum)
  reference  range_gamma.npz      generated with E^-2  (the right spectrum)
If reweighting works, the weighted test sample reproduces the reference sample,
including the absolute normalisation (Sum of weights).

  E1  weighted <dsigma/dx>, <dsigma/dy>, <dsigma/dQ^2> and dN/dE_nu:
      test (points), reference (points), spline averaged over the reference's
      weighted energy spectrum (line), ratio panel test/reference
  E2  dsigma/dxdy from the GammaDIS override (what the Weighter uses) vs plain
      DISFromSpline at the same (E, x, y, Q^2): must lie on the diagonal
  E3  distribution of log10(weight) for test and reference (no NaN / zero weights)

Usage:
    python plot_gamma_dis_reweighting.py --samples output/gamma_dis_samples \
        --outdir output/gamma_dis_plots/reweighting
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gamma_dis_common as G

VARS = {
    "x":  dict(label=r"$x$", dens=r"$\langle d\sigma/dx\rangle_\Phi$  [arb. units]", log=True),
    "y":  dict(label=r"$y$", dens=r"$\langle d\sigma/dy\rangle_\Phi$  [arb. units]", log=False),
    "Q2": dict(label=r"$Q^2$  [GeV$^2$]",
               dens=r"$\langle d\sigma/dQ^2\rangle_\Phi$  [arb. units]", log=True),
    "E":  dict(label=r"Neutrino Energy  $E_\nu$  [GeV]",
               dens=r"weighted rate  $dN/dE_\nu$  [arb. units]", log=True),
}
STY = {"test": ("o", G.COLORS["test"], "Test: generated $E^{-1}$, reweighted to $E^{-2}$"),
       "ref":  ("s", G.COLORS["ref"],  "Reference: generated $E^{-2}$")}


def vals(s, v):
    return {"x": s["x"], "y": s["y"], "Q2": G.q2_of(s), "E": s["E_nu"]}[v]


def make_bins(arrs, v, n):
    a = np.concatenate([x[np.isfinite(x) & (x > 0)] for x in arrs])
    lo, hi = np.percentile(a, 0.2), np.percentile(a, 99.8)
    if v == "E":
        lo, hi = a.min(), a.max() * 1.0001
    return np.logspace(np.log10(lo), np.log10(hi), n + 1) if VARS[v]["log"] \
        else np.linspace(lo, hi, n + 1)


def e1(samples, truth, var, outdir, nbins):
    bins = make_bins([vals(s, var) for s in samples.values()], var, nbins)
    w = np.diff(bins)
    cen = np.sqrt(bins[1:] * bins[:-1]) if VARS[var]["log"] else 0.5 * (bins[1:] + bins[:-1])
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.2, 8.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    dens = {}
    for key, s in samples.items():
        wt = np.where(np.isfinite(s["weight"]) & (s["weight"] > 0), s["weight"], 0.0)
        h = np.histogram(vals(s, var), bins, weights=wt)[0]
        e = np.sqrt(np.histogram(vals(s, var), bins, weights=wt ** 2)[0])
        dens[key] = (h / w, e / w)
        mk, col, lab = STY[key]
        ax.errorbar(cen, h / w, yerr=e / w, fmt=mk, ms=7, color=col,
                    mfc=col if key == "test" else "white", mec=col, mew=1.4,
                    capsize=2, alpha=0.9, label=f"{lab}   ($\\Sigma w$ = {wt.sum():.3g})")
    if var != "E":
        ref = samples["ref"]
        M = float(np.median(ref["target_mass"]))
        wref = np.where(np.isfinite(ref["weight"]) & (ref["weight"] > 0), ref["weight"], 0.0)
        comps = G.energy_components(ref["E_nu"], wref)
        p = truth.expected_1d(var, bins, comps, M)
        curve = p * wref.sum() / w
        ax.plot(np.repeat(bins, 2)[1:-1], np.repeat(curve, 2), "-",
                color=G.COLORS["theory"], lw=2.2,
                label="Spline, averaged over the weighted spectrum")
    (dt, et), (dr, er) = dens["test"], dens["ref"]
    ok = dr > 0
    axr.errorbar(cen[ok], dt[ok] / dr[ok],
                 yerr=np.sqrt((et[ok] / dr[ok]) ** 2 + (dt[ok] * er[ok] / dr[ok] ** 2) ** 2),
                 fmt="o", ms=6, color=G.COLORS["test"], capsize=2)
    axr.axhline(1.0, color=G.COLORS["theory"], lw=1.4)
    axr.set_ylim(0.6, 1.4)
    axr.set_ylabel("test / reference", fontsize=13)
    if VARS[var]["log"]:
        ax.set_xscale("log")
    if var != "y":
        ax.set_yscale("log")
    ax.set_ylabel(VARS[var]["dens"], fontsize=14)
    axr.set_xlabel(VARS[var]["label"], fontsize=15)
    name = {"x": "x", "y": "y", "Q2": r"$Q^2$", "E": r"$E_\nu$"}[var]
    G.title(ax, f"Reweighting Closure: {name}")
    G.legend(ax)
    G.style(ax); G.style(axr)
    G.savefig(fig, outdir, f"reweight_{var}.png")
    plt.close(fig)


def e2(samples, outdir):
    xo = np.concatenate([s["xs_override"] for s in samples.values()])
    xb = np.concatenate([s["xs_base"] for s in samples.values()])
    E = np.concatenate([s["E_nu"] for s in samples.values()])
    ok = (xb > 0) & (xo > 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2))
    a1.scatter(xb[ok], xo[ok], s=10, color=G.COLORS["gamma"], alpha=0.6)
    lim = [xb[ok].min() * 0.8, xb[ok].max() * 1.25]
    a1.plot(lim, lim, "--", color=G.COLORS["theory"], lw=1.6, label="identical")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"DISFromSpline  $d^2\sigma/dx\,dy$", fontsize=14)
    a1.set_ylabel(r"GammaDISFromSpline override  $d^2\sigma/dx\,dy$", fontsize=14)
    G.title(a1, "Cross Section Used by the Weighter")
    G.legend(a1); G.style(a1)
    rel = np.abs(xo[ok] / xb[ok] - 1.0)
    a2.scatter(E[ok], np.clip(rel, 1e-17, None), s=10, color=G.COLORS["gamma"], alpha=0.6)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel(r"Neutrino Energy  $E_\nu$  [GeV]", fontsize=15)
    a2.set_ylabel("|override / base - 1|", fontsize=14)
    G.title(a2, "Relative Difference")
    a2.text(0.03, 0.95, f"worst = {rel.max():.1e}   ({ok.sum()} events, "
            f"{(~ok).sum()} with zero/NaN)", transform=a2.transAxes, fontsize=12,
            va="top", bbox=dict(fc="white", alpha=0.9, ec="none"))
    G.style(a2)
    G.savefig(fig, outdir, "reweight_xs_override_vs_base.png")
    plt.close(fig)


def e3(samples, outdir):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for key, s in samples.items():
        wt = s["weight"]
        good = np.isfinite(wt) & (wt > 0)
        mk, col, lab = STY[key]
        ax.hist(np.log10(wt[good]), bins=50, histtype="step", lw=2.2, color=col,
                label=f"{lab}  ({(~good).sum()} NaN/zero)")
    ax.set_xlabel(r"$\log_{10}$ (event weight)", fontsize=15)
    ax.set_ylabel("events", fontsize=14)
    G.title(ax, "Event Weight Distribution")
    G.legend(ax); G.style(ax)
    G.savefig(fig, outdir, "reweight_weights.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--outdir", default="output/gamma_dis_plots/reweighting")
    ap.add_argument("--nbins", type=int, default=30)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    samples = {"test": G.load_sample(os.path.join(args.samples, "reweight_gamma.npz")),
               "ref": G.load_sample(os.path.join(args.samples, "range_gamma.npz"))}
    for k, s in samples.items():
        bad = ~(np.isfinite(s["weight"]) & (s["weight"] > 0))
        if bad.any():
            print(f"WARNING: {k}: {bad.sum()} events with NaN/zero weight "
                  "(counted as weight 0 in E1, listed in E3)")
    truth = G.SplineTruth(cache=args.cache or os.path.join(args.samples, "theory_cache.npz"))
    for var in VARS:
        e1(samples, truth, var, args.outdir, args.nbins)
    e2(samples, args.outdir)
    e3(samples, args.outdir)


if __name__ == "__main__":
    main()
