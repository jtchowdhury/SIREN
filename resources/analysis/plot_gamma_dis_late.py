#!/usr/bin/env python3
"""
plot_gamma_dis_late.py -- late-developing event showers drawn from the gamma-mixture
model. For each event of a ccnc_mono sample (same boosted PYTHIA hadron lists as
SIREN), the showers are redrawn --n-draws times; draws with a late peak (a local
maximum beyond --x-late above --min-height of the main peak) are kept, and the
--n-select with the most late light are plotted. Legends say they are selected.

    python plot_gamma_dis_late.py --samples output/gamma_dis_samples --enu 1e4 1e5
"""
import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import find_peaks

import sys
import gamma_dis_common as G
sys.path.insert(0, os.path.join(G.SIREN_ROOT, "python"))
import hadronic_showers as hs                                   # noqa: E402

COL = {"cc": "#4c78a8", "nc": "#f58518"}


def event_draw(model, s, e, ch, x, rng):
    h0, h1 = s["had_off"][e], s["had_off"][e + 1]
    parts = [(int(s["had_pid"][i]), G.e_kin(s["had_pid"][i], float(s["had_p"][i, 0])))
             for i in range(h0, h1)]
    if ch == "cc":
        parts.append((11, G.e_kin(11, float(s["lep"][e, 0]))))
    y = np.zeros_like(x)
    for pid, E in parts:
        if E > 0 and model.has(pid):
            y += model.sample(pid, E, rng, x=x).photons
    return y


def late_score(x, y, x_late, min_height):
    a = hs._trapz(y, x)
    if a <= 0:
        return -1.0
    yn = y / a
    pk, _ = find_peaks(yn)
    late = [p for p in pk if x[p] > x_late and yn[p] > min_height * yn.max()]
    if not late:
        return -1.0
    return hs._trapz(yn[x > x_late], x[x > x_late])


def plot(x, groups, path, xlim, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for profs, col, lab in groups:
        for i, y in enumerate(profs):
            ax.plot(x, y / hs._trapz(y, x), color=col, lw=1.0, alpha=0.8,
                    label=lab if i == 0 else None)
    ax.set_xlim(*xlim)
    ax.set_xlabel(hs.XLABEL, fontsize=16)
    ax.set_ylabel(hs.YLABEL, fontsize=16)
    ax.tick_params(labelsize=14)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=14, title=title, title_fontsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--model", default=os.path.join(
        G.SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.npz"))
    ap.add_argument("--outdir", default="output/gamma_dis_plots/talk")
    ap.add_argument("--enu", type=float, nargs="+", default=[1e4, 1e5])
    ap.add_argument("--channels", nargs="+", default=["cc", "nc"])
    ap.add_argument("--n-select", type=int, default=5)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--x-late", type=float, default=750.0)
    ap.add_argument("--min-height", type=float, default=0.3)
    ap.add_argument("--xlim", type=float, nargs=2, default=[0, 2000])
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*outside the trained range.*")
    rng = np.random.default_rng(args.seed)
    model = hs.load_model(args.model)
    x = np.arange(2.5, 3000.0, 5.0)

    for Enu in args.enu:
        el = G.elabel(Enu)
        groups = []
        for ch in args.channels:
            s = G.load_sample(os.path.join(args.samples, f"ccnc_mono_E{Enu:.0e}_{ch}.npz"))
            n_ev = len(s["E_nu"])
            found = []
            for e in range(n_ev):
                for _ in range(args.n_draws):
                    y = event_draw(model, s, e, ch, x, rng)
                    sc = late_score(x, y, args.x_late, args.min_height)
                    if sc > 0:
                        found.append((sc, y))
                print(f"  {el} {ch}: event {e + 1}/{n_ev}, late draws so far {len(found)}", end="\r")
            print()
            tot = n_ev * args.n_draws
            print(f"{el} {ch.upper()}: {len(found)} of {tot} draws have a late peak "
                  f"({100 * len(found) / tot:.2f}%)")
            found.sort(key=lambda t: -t[0])
            sel = [y for _, y in found[:args.n_select]]
            if not sel:
                continue
            lab = f"{el} {ch.upper()} ({len(sel)} selected)"
            plot(x, [(sel, COL[ch], lab)],
                 os.path.join(args.outdir, f"late_E{Enu:.0e}_{ch}.png"), args.xlim,
                 "Sampled Gamma Shower, late-developing")
            groups.append((sel, COL[ch], lab))
        if groups:
            plot(x, groups, os.path.join(args.outdir, f"late_E{Enu:.0e}_ccnc.png"),
                 args.xlim, "Sampled Gamma Shower, late-developing")


if __name__ == "__main__":
    main()
