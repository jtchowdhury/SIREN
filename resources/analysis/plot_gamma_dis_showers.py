#!/usr/bin/env python3
"""
plot_gamma_dis_showers.py -- section C: do the subshowers coming OUT of SIREN
match the shower model they were supposed to be drawn from?

C1  per particle. For every parent particle in an event, its output shower is
    built from the stored components:  f(z) = Sum_k amp_k * Gamma(z; alpha_k, beta_k).
    It is compared with --n-draws showers drawn from the Python shower model at the
    SAME species and parent energy (hadron<i>_pid / hadron<i>_E, or the electron's
    lep_E). L2 and KS are averaged over the draws, then over the particles in each
    energy bin, and plotted vs energy -- one figure per species plus all species.
C2  composite. Each event's composite output shower (sum over all its subshowers)
    vs --n-draws composites built by redrawing every parent from the model; vs E_had.

Metric (identical to the earlier event-vs-G4 plots): relative
L2 = Sum (ref - test)^2 / Sum ref^2 with ref = the model draw, and KS = max gap of
the normalised cumulative profiles. The dashed floor is the same metric between
two independent model draws: showers fluctuate, so a perfect implementation sits
ON the floor, not at zero.

Usage:
    python plot_gamma_dis_showers.py --samples output/gamma_dis_samples \
        --sample range_gamma.npz --max-events 500 --n-draws 100 \
        --outdir output/gamma_dis_plots/showers
"""
import os
import sys
import csv
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gamma_dis_common as G

sys.path.insert(0, os.path.join(G.SIREN_ROOT, "python"))
import hadronic_showers as hs                                    # noqa: E402

META = {"out":   ("GammaDISFromSpline output vs model", "#F08070"),
        "floor": ("Model vs model (empirical floor)", "#444444")}


def l2_ks(ref, test):
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    denom = float(np.dot(ref, ref))
    if denom <= 0:
        return np.nan, np.nan
    l2 = float(np.sum((ref - test) ** 2) / denom)
    rp, tp = np.clip(ref, 0, None), np.clip(test, 0, None)
    rs, ts = rp.sum(), tp.sum()
    ks = float(np.max(np.abs(np.cumsum(rp) / rs - np.cumsum(tp) / ts))) \
        if rs > 0 and ts > 0 else np.nan
    return l2, ks


def mean_vs(ref_draws, test):
    m = np.array([l2_ks(r, test) for r in ref_draws])
    return np.nanmean(m[:, 0]), np.nanmean(m[:, 1])


def binned(E, vals, nbins=12, min_count=3):
    E = np.asarray(E, float)
    edges = np.logspace(np.log10(E.min()), np.log10(E.max() * 1.0001), nbins + 1)
    idx = np.digitize(E, edges) - 1
    out = []
    for b in range(nbins):
        sel = idx == b
        if sel.sum() >= min_count:
            out.append([np.sqrt(edges[b] * edges[b + 1])] +
                       [float(np.nanmean(np.asarray(v)[sel])) for v in vals])
    return np.array(out)


def two_panel(rows, ttl, xlabel, outdir, fname):
    """rows: columns E, l2_out, ks_out, l2_floor, ks_floor  (style of event_vs_g4)."""
    if len(rows) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)
    for ax, (io, ifl) in ((ax1, (1, 3)), (ax2, (2, 4))):
        for key, col in (("out", io), ("floor", ifl)):
            label, color = META[key]
            ok = np.isfinite(rows[:, col]) & (rows[:, col] > 0)
            if key == "floor":
                ax.plot(rows[ok, 0], rows[ok, col], "--", color=color, lw=2.0,
                        alpha=0.9, label=label)
            else:
                ax.plot(rows[ok, 0], rows[ok, col], "-", color=color, lw=2.6,
                        marker="o", ms=9, markeredgecolor="white",
                        markeredgewidth=1.3, alpha=0.85, label=label)
        ax.set_yscale("log")
        G.style(ax)
        G.legend(ax)
    ax1.set_ylabel(r"$L_2$ ($\sum(\mathrm{model}-\mathrm{output})^2/\sum\mathrm{model}^2$)",
                   fontsize=13)
    ax2.set_ylabel("KS statistic", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_xlabel(xlabel, fontsize=15)
    G.title(ax1, ttl)
    G.savefig(fig, outdir, fname)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--sample", default="range_gamma.npz")
    ap.add_argument("--model", default=os.path.join(
        G.SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.npz"))
    ap.add_argument("--max-events", type=int, default=500)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="output/gamma_dis_plots/showers")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*outside the trained range.*")

    s = G.load_sample(os.path.join(args.samples, args.sample))
    model = hs.load_model(args.model)
    z = model.z_centers(211)
    rng = np.random.default_rng(args.seed)
    nd = args.n_draws

    part_rows, comp_rows = [], []        # (species, E, l2o, kso, l2f, ksf) / (E_had, ...)
    n_ev = min(args.max_events, len(s["E_nu"]))
    for e in range(n_ev):
        a, b = s["sub_off"][e], s["sub_off"][e + 1]
        h0 = s["had_off"][e]
        parents = s["sub_parent"][a:b]
        comp_out = np.zeros_like(z)
        comp_draws = np.zeros((nd + 1, len(z)))
        for p in np.unique(parents):
            if p >= 0:
                pid, E = int(s["had_pid"][h0 + p]), float(s["had_p"][h0 + p, 0])
            else:
                pid, E = 11, float(s["lep"][e, 0])
            if not model.has(pid) or E <= 0:
                continue
            sel = np.arange(a, b)[parents == p]
            out = sum(s["sub_amp"][k] * hs._kernel(z, s["sub_alpha"][k], s["sub_beta"][k])
                      for k in sel)
            draws = np.array([model.sample(pid, E, rng, x=z).photons for _ in range(nd + 1)])
            l2o, kso = mean_vs(draws[:nd], out)
            l2f, ksf = mean_vs(draws[:nd], draws[nd])
            part_rows.append((hs.PID_TO_NAME[pid], E, l2o, kso, l2f, ksf))
            comp_out += out
            comp_draws += draws
        if comp_out.sum() > 0:
            l2o, kso = mean_vs(comp_draws[:nd], comp_out)
            l2f, ksf = mean_vs(comp_draws[:nd], comp_draws[nd])
            comp_rows.append((float(s["blob"][e, 0]), l2o, kso, l2f, ksf))
        print(f"event {e + 1}/{n_ev}", end="\r")
    print()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "showers_per_particle.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["species", "E_GeV", "l2_out", "ks_out", "l2_floor", "ks_floor"])
        w.writerows(part_rows)
    with open(os.path.join(args.outdir, "showers_composite.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["E_had_GeV", "l2_out", "ks_out", "l2_floor", "ks_floor"])
        w.writerows(comp_rows)

    names = np.array([r[0] for r in part_rows])
    vals = np.array([r[1:] for r in part_rows], float)
    xlabel = r"Parent Particle Energy  $E$  [GeV]"
    two_panel(binned(vals[:, 0], vals[:, 1:].T), "SIREN Output vs Shower Model: All Species",
              xlabel, args.outdir, "showers_all_species.png")
    for name in sorted(set(names)):
        sel = names == name
        if sel.sum() < 6:
            continue
        latex = hs.SPECIES_LATEX.get(name, name)
        two_panel(binned(vals[sel, 0], vals[sel, 1:].T, nbins=8),
                  f"SIREN Output vs Shower Model: {latex}", xlabel,
                  args.outdir, f"showers_{name}.png")

    cv = np.array(comp_rows, float)
    if len(cv):
        two_panel(binned(cv[:, 0], cv[:, 1:].T), "SIREN Output vs Shower Model: Event Composite",
                  r"Hadronic Energy  $E_\mathrm{had}$  [GeV]", args.outdir, "showers_composite.png")


if __name__ == "__main__":
    main()
