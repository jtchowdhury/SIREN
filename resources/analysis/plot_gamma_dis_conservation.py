#!/usr/bin/env python3
"""
plot_gamma_dis_conservation.py -- section B: is 4-momentum conserved per event?

  B1  total event:  | p_nu + p_target - (Sum p_hadrons + p_lepton) | / E_nu   vs E_nu
  B2  diagnostic:   | Sum p_hadrons - p_blob | / E_had                       vs E_had

One marker colour per component (E, px, py, pz). Values at exact zero are drawn
at the floor 1e-17 so they stay on the log axis. Uses every *_gamma.npz sample.

Usage:
    python plot_gamma_dis_conservation.py --samples output/gamma_dis_samples \
        --outdir output/gamma_dis_plots/conservation
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gamma_dis_common as G

COMP = [("E", "#222222", "o"), (r"p_x", "#F08070", "s"),
        (r"p_y", "#4682B4", "^"), (r"p_z", "#9B4DCA", "D")]
FLOOR = 1e-17


def hadron_sums(s):
    off, P = s["had_off"], s["had_p"]
    return np.array([P[off[i]:off[i + 1]].sum(0) for i in range(len(off) - 1)])


def draw(ax, xv, R, xlabel, ylabel, ttl):
    for k, (name, col, mk) in enumerate(COMP):
        ax.scatter(xv, np.clip(R[:, k], FLOOR, None), s=14, color=col, marker=mk,
                   alpha=0.6, label=rf"${name}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=14)
    G.title(ax, ttl)
    worst = np.nanmax(R)
    ax.text(0.03, 0.95, f"worst = {worst:.1e}   ({len(xv)} events)",
            transform=ax.transAxes, fontsize=12, va="top",
            bbox=dict(fc="white", alpha=0.9, ec="none"))
    G.legend(ax, loc="lower right")
    G.style(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--outdir", default="output/gamma_dis_plots/conservation")
    args = ap.parse_args()

    E_nu, R1, E_had, R2 = [], [], [], []
    files = sorted(glob.glob(os.path.join(args.samples, "*_gamma.npz")))
    if not files:
        raise SystemExit(f"no *_gamma.npz in {args.samples}")
    for f in files:
        s = G.load_sample(f)
        H = hadron_sums(s)
        p_target = np.zeros_like(s["p_nu"]); p_target[:, 0] = s["target_mass"]
        r1 = s["p_nu"] + p_target - (H + s["lep"])
        R1.append(np.abs(r1) / s["E_nu"][:, None]); E_nu.append(s["E_nu"])
        r2 = H - s["blob"]
        R2.append(np.abs(r2) / s["blob"][:, :1]); E_had.append(s["blob"][:, 0])
        print(f"{os.path.basename(f):28s} worst total {np.abs(r1).max() / 1:.2e} GeV")
    E_nu, R1 = np.concatenate(E_nu), np.concatenate(R1)
    E_had, R2 = np.concatenate(E_had), np.concatenate(R2)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    draw(ax, E_nu, R1, r"Neutrino Energy  $E_\nu$  [GeV]",
         r"$|p_\nu + p_\mathrm{target} - (\sum p_\mathrm{had} + p_\ell)|\,/\,E_\nu$",
         "Event 4-Momentum Conservation")
    G.savefig(fig, args.outdir, "conservation_total_vs_Enu.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    draw(ax, E_had, R2, r"Hadronic Energy  $E_\mathrm{had}$  [GeV]",
         r"$|\sum p_\mathrm{had} - p_\mathrm{blob}|\,/\,E_\mathrm{had}$",
         "Hadron Breakdown vs Hadronic System")
    G.savefig(fig, args.outdir, "conservation_hadrons_vs_Ehad.png"); plt.close(fig)


if __name__ == "__main__":
    main()
