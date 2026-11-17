#!/usr/bin/env python3
"""
validate_hadronization.py -- GammaDISFromSpline.Hadronize (Pythia hadronization of
a quark-diquark string) against full PYTHIA DIS events.

For library events (full PYTHIA nu_e CC DIS on a proton), Hadronize is called with
the event's own hadronic 4-momentum and Q, so both sides have the same W and the
same boost. Both hadron lists are cut to the 20 most energetic particles, as the
library stores them. Compared, in ranges of W:
  - number of particles with more than 2% of E_had
  - energy share of the leading particle
  - energy share of pi0, photons, charged pions, kaons, nucleons

Writes hadronization_W<lo>-<hi>.png (distributions), hadronization_means.png
(means vs W) and hadronization_summary.csv.

    python validate_hadronization.py --n 20000 --outdir output/gamma_dis_plots/hadronization
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import siren
import gamma_dis_common as G
from gamma_dis_samples import load_xs, PYTHIA

PT = siren.dataclasses.Particle.ParticleType
GROUPS = {"pi0": (111,), "photon": (22,), "charged pion": (211,), "kaon": (321, 310, 130, 311),
          "nucleon": (2212, 2112)}
COL = {"lib": "#4682B4", "new": "#F08070"}
LAB = {"lib": "full PYTHIA DIS (library)", "new": "string hadronization (GammaDIS)"}
W_RANGES = [(2, 5), (5, 15), (15, 50), (50, 150), (150, 1e9)]


def observables(ids, E, E_had):
    o = np.argsort(-E)[:20]
    ids, E = np.abs(ids[o]), E[o]
    out = {"n2": float((E > 0.02 * E_had).sum()), "lead": float(E.max() / E_had)}
    for name, codes in GROUPS.items():
        out[name] = float(E[np.isin(ids, codes)].sum() / E_had)
    return out


def wlabel(lo, hi):
    return f"W = {lo:g}-{hi:g} GeV" if hi < 1e9 else f"W > {lo:g} GeV"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=PYTHIA)
    ap.add_argument("--n", type=int, default=20000, help="library events to use")
    ap.add_argument("--outdir", default="output/gamma_dis_plots/hadronization")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with np.load(os.path.expanduser(args.library)) as d:
        ph, pl, Enu = d["p_had"], d["p_lepton"], d["E_nu"]
        pids, en = d["top20_pids"], d["top20_energies"]
    W = np.sqrt(np.maximum(ph[:, 0] ** 2 - (ph[:, 1:] ** 2).sum(1), 0.0))
    # the library neutrino travels along +z
    Q2 = -((Enu - pl[:, 0]) ** 2 - pl[:, 1] ** 2 - pl[:, 2] ** 2 - (Enu - pl[:, 3]) ** 2)
    Q = np.sqrt(np.maximum(Q2, 0.0))

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(W), min(args.n, len(W)), replace=False)
    xs = load_xs(True)[0]
    random = siren.utilities.SIREN_random(args.seed)

    rows, worst = [], 0.0
    for j, i in enumerate(idx):
        if W[i] < W_RANGES[0][0]:
            continue
        had = xs.Hadronize([float(v) for v in ph[i]], float(Q[i]), PT.NuE, PT.PPlus, random)
        ids = np.array([h[0] for h in had], int)
        p4 = np.array([h[1] for h in had], float)
        worst = max(worst, float(np.abs(p4.sum(0) - ph[i]).max() / ph[i, 0]))
        lep = np.isin(np.abs(ids), (11, 12, 13, 14, 15, 16))
        m = pids[i] != 0
        rows.append((W[i], observables(pids[i][m], en[i][m], ph[i, 0]),
                     observables(ids[~lep], p4[~lep, 0], ph[i, 0])))
        if j % 500 == 0:
            print(f"event {j}/{len(idx)}", end="\r")
    print(f"\n{len(rows)} events; worst |Sum(hadrons) - p_had| / E_had = {worst:.1e}")

    Ws = np.array([r[0] for r in rows])
    keys = list(rows[0][1].keys())
    val = {s: {k: np.array([r[c][k] for r in rows]) for k in keys} for s, c in (("lib", 1), ("new", 2))}

    # distributions per W range
    panels = [("n2", "particles with > 2% of $E_\\mathrm{had}$", np.arange(-0.5, 21.5, 1.0)),
              ("lead", "energy share of the leading particle", np.linspace(0, 1, 26)),
              ("pi0", "energy share of $\\pi^0$", np.linspace(0, 1, 26))]
    for lo, hi in W_RANGES:
        sel = (Ws >= lo) & (Ws < hi)
        if sel.sum() < 50:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        for ax, (k, xl, bins) in zip(axes, panels):
            for s in ("lib", "new"):
                v = val[s][k][sel]
                ax.hist(v, bins=bins, density=True, histtype="step", lw=2.2, color=COL[s],
                        label=f"{LAB[s]}   mean {v.mean():.2f}")
            ax.set_xlabel(xl, fontsize=14)
            ax.set_ylabel("normalized events", fontsize=13)
            G.legend(ax)
            G.style(ax)
        G.title(axes[1], f"Hadronization: {wlabel(lo, hi)}  ({sel.sum()} events)")
        tag = f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}-inf"
        G.savefig(fig, args.outdir, f"hadronization_W{tag}.png")
        plt.close(fig)

    # means vs W
    edges = np.logspace(np.log10(W_RANGES[0][0]), np.log10(Ws.max() * 1.001), 13)
    cen = np.sqrt(edges[1:] * edges[:-1])
    b = np.digitize(Ws, edges) - 1
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), sharex=True)
    names = {"n2": "particles with > 2% of $E_\\mathrm{had}$", "lead": "leading-particle share"}
    with open(os.path.join(args.outdir, "hadronization_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "W_GeV", "n", "library_mean", "string_mean"])
        for ax, k in zip(axes.ravel(), keys):
            for s in ("lib", "new"):
                mean = np.array([val[s][k][b == i].mean() if (b == i).sum() >= 20 else np.nan
                                 for i in range(len(cen))])
                err = np.array([val[s][k][b == i].std() / np.sqrt((b == i).sum())
                                if (b == i).sum() >= 20 else np.nan for i in range(len(cen))])
                ax.errorbar(cen, mean, yerr=err, fmt="o-" if s == "new" else "s-", ms=6, lw=2,
                            color=COL[s], mfc=COL[s] if s == "new" else "white", capsize=2,
                            label=LAB[s])
            for i in range(len(cen)):
                if (b == i).sum() >= 20:
                    w.writerow([k, f"{cen[i]:.4g}", int((b == i).sum()),
                                f"{val['lib'][k][b == i].mean():.4f}", f"{val['new'][k][b == i].mean():.4f}"])
            ax.set_xscale("log")
            ax.set_ylabel(names.get(k, f"energy share: {k}"), fontsize=13)
            G.style(ax)
        for ax in axes[1]:
            ax.set_xlabel(r"Hadronic invariant mass  $W$  [GeV]", fontsize=14)
        G.legend(axes[0, 0])
        axes[1, 3].axis("off")
        G.title(axes[0, 1], "Hadronization: Means vs W")
        G.savefig(fig, args.outdir, "hadronization_means.png")
        plt.close(fig)
    print("wrote", os.path.join(args.outdir, "hadronization_summary.csv"))


if __name__ == "__main__":
    main()
