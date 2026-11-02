#!/usr/bin/env python3
"""
plot_gamma_dis_kinematics.py -- section A: are the sampled DIS kinematics right?

For every monochromatic energy and for the energy range, compares
GammaDISFromSpline events, DISFromSpline events, and the CSMS spline itself.

  A1-A3  (1/sigma) dsigma/dx vs x, dsigma/dy vs y, dsigma/dQ^2 vs Q^2
         (events as points with Poisson errors, spline as a line, ratio panel)
  A4-A6  2D maps in (x, y), (x, Q^2), (y, Q^2):
         rows = GammaDIS, DISFromSpline; columns = events, spline, pull
  A7-A8  <y> and <Q^2> vs E_nu from the range samples, with the spline curve

For the range samples the spline "truth" is the average of the normalised
dsigma/dx dy over the energies that were actually generated (unweighted events
follow the generation spectrum).

Usage:
    python plot_gamma_dis_kinematics.py --samples output/gamma_dis_samples \
        --outdir output/gamma_dis_plots/kinematics
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import gamma_dis_common as G

VARS = {
    "x":  dict(label=r"$x$", dens=r"$(1/\sigma)\,d\sigma/dx$", log=True, ylog=True),
    "y":  dict(label=r"$y$", dens=r"$(1/\sigma)\,d\sigma/dy$", log=False, ylog=False),
    "Q2": dict(label=r"$Q^2$  [GeV$^2$]", dens=r"$(1/\sigma)\,d\sigma/dQ^2$  [GeV$^{-2}$]",
               log=True, ylog=True),
}
PLANES = [("x", "y"), ("x", "Q2"), ("y", "Q2")]


def values(s, var):
    return {"x": s["x"], "y": s["y"], "Q2": G.q2_of(s)}[var]


def make_bins(arrs, var, n):
    v = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrs])
    lo, hi = np.percentile(v, 0.1), np.percentile(v, 99.9)
    if VARS[var]["log"]:
        return np.logspace(np.log10(lo), np.log10(hi), n + 1)
    return np.linspace(max(lo, 0.0), min(hi, 1.0), n + 1)


def marginal(samples, truth, comps, M, var, tag, ftag, outdir, nbins):
    """samples: {'gamma': s, 'dis': s}; one figure with a ratio panel."""
    bins = make_bins([values(s, var) for s in samples.values()], var, nbins)
    w = np.diff(bins)
    centers = np.sqrt(bins[1:] * bins[:-1]) if VARS[var]["log"] else 0.5 * (bins[1:] + bins[:-1])
    p_exp = truth.expected_1d(var, bins, comps, M)
    dens_exp = p_exp / w

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.2, 8.6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    fine = np.repeat(bins, 2)[1:-1]
    ax.plot(fine, np.repeat(dens_exp, 2), "-", color=G.COLORS["theory"], lw=2.2,
            label=G.LABELS["theory"])
    for key, marker in (("gamma", "o"), ("dis", "s")):
        if key not in samples:
            continue
        v = values(samples[key], var)
        N = np.isfinite(v).sum()
        n = np.histogram(v, bins)[0]
        d, e = n / (N * w), np.sqrt(n) / (N * w)
        mfc = G.COLORS[key] if key == "gamma" else "white"
        ax.errorbar(centers, d, yerr=e, fmt=marker, ms=7, color=G.COLORS[key],
                    mfc=mfc, mec=G.COLORS[key], mew=1.4, capsize=2, alpha=0.9,
                    label=f"{G.LABELS[key]}  ({N} events)")
        ok = dens_exp > 0
        axr.errorbar(centers[ok], d[ok] / dens_exp[ok], yerr=e[ok] / dens_exp[ok],
                     fmt=marker, ms=6, color=G.COLORS[key], mfc=mfc, mec=G.COLORS[key],
                     mew=1.2, capsize=2, alpha=0.9)
    axr.axhline(1.0, color=G.COLORS["theory"], lw=1.4)
    axr.set_ylim(0.7, 1.3)
    if VARS[var]["log"]:
        ax.set_xscale("log")
    if VARS[var]["ylog"]:
        ax.set_yscale("log")
    ax.set_ylabel(VARS[var]["dens"], fontsize=14)
    axr.set_ylabel("events / spline", fontsize=13)
    axr.set_xlabel(VARS[var]["label"], fontsize=15)
    var_txt = {"x": "x", "y": "y", "Q2": r"$Q^2$"}[var]
    G.title(ax, f"Differential Cross Section vs {var_txt}  ({tag})")
    G.legend(ax)
    G.style(ax); G.style(axr)
    G.savefig(fig, outdir, f"dsigma_d{var}_{ftag}.png")
    plt.close(fig)


def plane(samples, truth, comps, M, va, vb, tag, ftag, outdir, nbins):
    ba = make_bins([values(s, va) for s in samples.values()], va, nbins)
    bb = make_bins([values(s, vb) for s in samples.values()], vb, nbins)
    p_exp = truth.expected_2d(va, vb, ba, bb, comps, M)
    area = np.outer(np.diff(ba), np.diff(bb))
    keys = [k for k in ("gamma", "dis") if k in samples]
    fig, axes = plt.subplots(len(keys), 3, figsize=(18, 5.6 * len(keys)), squeeze=False)
    dens_exp = p_exp / area
    vmin = dens_exp[dens_exp > 0].min() if np.any(dens_exp > 0) else 1e-10
    norm = LogNorm(vmin=max(vmin, dens_exp.max() * 1e-6), vmax=dens_exp.max())
    for r, key in enumerate(keys):
        a, b = values(samples[key], va), values(samples[key], vb)
        N = (np.isfinite(a) & np.isfinite(b)).sum()
        n = np.histogram2d(a, b, [ba, bb])[0]
        mu = N * p_exp
        pull = np.where(mu > 0, (n - mu) / np.sqrt(np.where(mu > 0, mu, 1)), np.nan)
        pull[(mu < 1) & (n == 0)] = np.nan          # empty, expected-empty bins
        panels = [(n / (N * area), "events", "viridis", norm),
                  (dens_exp, "spline", "viridis", norm),
                  (pull, "pull  (events - spline)/$\\sqrt{\\mathrm{spline}}$", "RdBu_r", None)]
        for c, (Z, what, cmap, nrm) in enumerate(panels):
            ax = axes[r, c]
            Zm = np.ma.masked_where(~np.isfinite(Z) | ((Z <= 0) & (nrm is not None)), Z)
            kw = dict(cmap=cmap, shading="flat")
            if nrm is not None:
                kw["norm"] = nrm
            else:
                kw.update(vmin=-4, vmax=4)
            im = ax.pcolormesh(ba, bb, Zm.T, **kw)
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("normalised density" if nrm is not None else "pull", fontsize=12)
            if VARS[va]["log"]:
                ax.set_xscale("log")
            if VARS[vb]["log"]:
                ax.set_yscale("log")
            ax.set_xlabel(VARS[va]["label"], fontsize=15)
            ax.set_ylabel(VARS[vb]["label"], fontsize=15)
            G.title(ax, G.LABELS["theory"] if c == 1 else f"{G.LABELS[key]}: {what.split('  ')[0]}")
            ax.tick_params(labelsize=12)
            if c == 2:
                ok = np.isfinite(pull)
                ax.text(0.03, 0.04, f"mean {np.nanmean(pull):+.2f}, rms {np.nanstd(pull):.2f}",
                        transform=ax.transAxes, fontsize=11,
                        bbox=dict(fc="white", alpha=0.85, ec="none"))
    short = {"x": "$x$", "y": "$y$", "Q2": "$Q^2$"}
    fig.suptitle(f"{short[vb]} vs {short[va]}  ({tag})",
                 fontsize=16, fontweight="bold")
    G.savefig(fig, outdir, f"plane_{va}_{vb}_{ftag}.png")
    plt.close(fig)


def means_vs_energy(samples, truth, M, outdir, nbins=10):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)
    Eall = np.concatenate([s["E_nu"] for s in samples.values()])
    edges = np.logspace(np.log10(Eall.min()), np.log10(Eall.max()), nbins + 1)
    Egrid = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), 14)
    a1.plot(Egrid, [truth.mean("y", E, M) for E in Egrid], "-",
            color=G.COLORS["theory"], lw=2.2, label=G.LABELS["theory"])
    a2.plot(Egrid, [truth.mean("Q2", E, M) for E in Egrid], "-",
            color=G.COLORS["theory"], lw=2.2, label=G.LABELS["theory"])
    for key, marker in (("gamma", "o"), ("dis", "s")):
        if key not in samples:
            continue
        s = samples[key]
        E, y, q2 = s["E_nu"], s["y"], G.q2_of(s)
        idx = np.digitize(E, edges) - 1
        cen, my, ey, mq, eq = [], [], [], [], []
        for b in range(nbins):
            sel = idx == b
            if sel.sum() < 5:
                continue
            cen.append(np.sqrt(edges[b] * edges[b + 1]))
            my.append(y[sel].mean()); ey.append(y[sel].std() / np.sqrt(sel.sum()))
            mq.append(q2[sel].mean()); eq.append(q2[sel].std() / np.sqrt(sel.sum()))
        mfc = G.COLORS[key] if key == "gamma" else "white"
        for ax, m, e in ((a1, my, ey), (a2, mq, eq)):
            ax.errorbar(cen, m, yerr=e, fmt=marker, ms=8, color=G.COLORS[key], mfc=mfc,
                        mec=G.COLORS[key], mew=1.4, capsize=2, label=G.LABELS[key])
    a1.set_ylabel(r"$\langle y \rangle$", fontsize=15)
    a2.set_ylabel(r"$\langle Q^2 \rangle$  [GeV$^2$]", fontsize=14)
    a2.set_xlabel(r"Neutrino Energy  $E_\nu$  [GeV]", fontsize=15)
    a2.set_xscale("log"); a2.set_yscale("log")
    G.title(a1, "Mean Inelasticity and Momentum Transfer vs Energy")
    for ax in (a1, a2):
        G.legend(ax); G.style(ax)
    G.savefig(fig, outdir, "means_vs_energy.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--outdir", default="output/gamma_dis_plots/kinematics")
    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--nbins2d", type=int, default=30)
    ap.add_argument("--cache", default=None, help="spline grid cache (default: <samples>/theory_cache.npz)")
    args = ap.parse_args()
    truth = G.SplineTruth(cache=args.cache or os.path.join(args.samples, "theory_cache.npz"))

    # monochromatic
    for gpath in sorted(glob.glob(os.path.join(args.samples, "mono_E*_gamma.npz"))):
        stem = os.path.basename(gpath)[len("mono_E"):-len("_gamma.npz")]
        E = float(stem)
        samples = {"gamma": G.load_sample(gpath)}
        dpath = gpath.replace("_gamma.npz", "_dis.npz")
        if os.path.exists(dpath):
            samples["dis"] = G.load_sample(dpath)
        M = float(np.median(samples["gamma"]["target_mass"]))
        tag, ftag = rf"$E_\nu$ = {G.elabel(E)}", f"E{E:.0e}"
        print(f"\n== monochromatic {tag} ==")
        for var in VARS:
            marginal(samples, truth, [(E, 1.0)], M, var, tag, ftag, args.outdir, args.nbins)
        for va, vb in PLANES:
            plane(samples, truth, [(E, 1.0)], M, va, vb, tag, ftag, args.outdir, args.nbins2d)

    # energy range
    rg = os.path.join(args.samples, "range_gamma.npz")
    if os.path.exists(rg):
        samples = {"gamma": G.load_sample(rg)}
        rd = os.path.join(args.samples, "range_dis.npz")
        if os.path.exists(rd):
            samples["dis"] = G.load_sample(rd)
        Eg = samples["gamma"]["E_nu"]
        M = float(np.median(samples["gamma"]["target_mass"]))
        comps = G.energy_components(Eg)            # unweighted: generation spectrum
        tag, ftag = rf"$E_\nu$ = {G.elabel(Eg.min())}–{G.elabel(Eg.max())}", "range"
        print(f"\n== range {tag} ({len(comps)} energy slices) ==")
        for var in VARS:
            marginal(samples, truth, comps, M, var, tag, ftag, args.outdir, args.nbins)
        for va, vb in PLANES:
            plane(samples, truth, comps, M, va, vb, tag, ftag, args.outdir, args.nbins2d)
        means_vs_energy(samples, truth, M, args.outdir)


if __name__ == "__main__":
    main()
