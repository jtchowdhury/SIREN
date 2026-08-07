"""
compare_sampled_vs_g4.py
========================
A different comparison from the per-event reconstruction plot.

Here we compare a shower our model GENERATED to a shower Geant4 produced:
  - draw a shower from the G4 data sample,
  - draw a shower from our sampled model (gamma mixture / single gamma / analytic),
  - pair them (random pairing -- they have no 1-to-1 correspondence),
  - score the pair with a plain L2 and a KS, and average over many pairs.

Three curves vs energy, one per model.  Because a generated shower and a G4
shower are both random draws, this measures "how far a typical generated shower
is from a typical real shower" -- it folds in both model bias AND the intrinsic
shower-to-shower scatter (so a perfect model would land at the G4-vs-G4 level).

Detector resolution: the G4 shower is blurred with a Gaussian (sigma = c/n * 2ns
~ 45 cm); the generated shower is left as-is (per Andy).  Generated yields are
recalibrated to the G4 total-light distribution, so L2 is a fair shape+yield
comparison and isn't swamped by a yield offset.

Same 2-panel layout / colours / labels as perevent_l2ks.

Run
---
    python compare_sampled_vs_g4.py --species pip
    # data + model: ../output/data ; plots: output/plots/result
"""

import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shower_gamma_model import (
    load_model, load_g4_library, ShowerSampler, NAME_TO_PID, PID_TO_NAME,
)
import compare_methods as C

# same colours as perevent_l2ks; labels reflect that these are SAMPLES, not fits
SAMPLED_META = {
    "analytic": ("Analytic (fixed shape)", "#4682B4"),   # steel blue
    "single":   ("Sampled Single Gamma",   "#BA55D3"),   # medium orchid
    "mixture":  ("Gamma Mixture Model",     "#FA8072"),   # salmon
}


def sampled_vs_g4(g4, gen, x, sigma_cm, rng):
    """Randomly pair generated showers with (blurred) G4 showers; return the
    mean per-pair relative L2 and mean KS (normalized-profile CDF gap)."""
    binw = float(x[1] - x[0])
    sig_bins = max(sigma_cm / binw, 1e-6)
    ng, ns = len(g4), len(gen)
    n = min(ng, ns)
    gi = rng.permutation(ng)[:n]
    si = rng.permutation(ns)[:n]
    l2s, kss = [], []
    for a, b in zip(gi, si):
        g = gaussian_filter1d(np.asarray(g4[a], float), sig_bins, mode="constant")
        s = np.asarray(gen[b], float)
        denom = float(np.dot(g, g))
        if denom <= 0:
            continue
        l2s.append(float(np.sum((g - s) ** 2)) / denom)
        gp = np.clip(g, 0.0, None); sp = np.clip(s, 0.0, None)
        gs, ss = gp.sum(), sp.sum()
        if gs > 0 and ss > 0:
            kss.append(float(np.max(np.abs(np.cumsum(gp) / gs - np.cumsum(sp) / ss))))
    return (float(np.mean(l2s)) if l2s else np.nan,
            float(np.mean(kss)) if kss else np.nan)


def run_species(interp, sampler, library, pid, n_sample, seed, outdir, sigma_cm):
    name = PID_TO_NAME.get(pid, str(pid))
    rng = np.random.default_rng(seed)
    rows = []
    for E in sorted(library[pid].keys()):
        x = library[pid][E]["z_centers"]
        ens = C.build_ensembles(interp, sampler, library, pid, E, x, n_sample,
                                rng, fix_yield=True)
        g4 = ens["G4"]
        for tag, form in (("M1", "analytic"), ("M2", "single"), ("M3", "mixture")):
            gen = ens[tag]
            if gen is None:
                continue
            l2, ks = sampled_vs_g4(g4, gen, x, sigma_cm, rng)
            rows.append(dict(species=name, E=E, form=form, l2=l2, ks=ks))
        print(f"  E={E:8.0f}  " +
              "  ".join(f"{r['form']}:L2={r['l2']:.2g},KS={r['ks']:.2g}"
                       for r in rows if r["E"] == E))

    C._write_l2ks_csv(rows, os.path.join(outdir, f"sampled_vs_g4_{name}.csv"))
    _plot(rows, name, outdir, sigma_cm)
    return rows


def _plot(rows, name, outdir, sigma_cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = C.SPECIES_LATEX.get(name, name)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)

    def draw(ax, val_key):
        for form in ("analytic", "single", "mixture"):
            label, color = SAMPLED_META[form]
            E, v = C._series(rows, form, val_key, match_key="form")
            if len(E) == 0:
                continue
            ax.plot(E, v, "-", color=color, lw=2.6, marker="o", ms=10,
                    markeredgecolor="white", markeredgewidth=1.4, alpha=0.85,
                    label=label)
        ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=12, length=6)

    draw(ax1, "l2")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$L_2$   $\sum_x(\mathrm{data}-\mathrm{model})^2/\sum_x\mathrm{data}^2$",
                   fontsize=13)
    ax1.set_title(f"Sampled Model Shower vs Geant4 Shower ({sp})\n"
                  rf"G4 smeared with Gaussian $\sigma={sigma_cm:.0f}$ cm  ($v=c/n$, 2 ns)",
                  fontsize=15, fontweight="bold", pad=10)
    ax1.legend(fontsize=12, framealpha=0.92, loc="best")

    draw(ax2, "ks")
    ax2.set_xscale("log")
    ax2.set_ylabel("KS statistic  (max CDF gap)", fontsize=14)
    ax2.set_xlabel("Shower Energy [GeV]", fontsize=15)

    fig.tight_layout()
    out = os.path.join(outdir, f"sampled_vs_g4_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output/data")
    ap.add_argument("--model", default=None,
                    help="prebuilt model pkl (default: <g4-dir>/shower_model.pkl)")
    ap.add_argument("--species", default="pip")
    ap.add_argument("--n-sample", type=int, default=1000,
                    help="generated showers per model per energy")
    ap.add_argument("--depth-res-cm", type=float, default=C.DEPTH_RES_CM,
                    help="detector depth resolution for the G4 blur (default ~45 cm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/result")
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
            print(f"skip '{sp}': no G4 files in {args.g4_dir}"); continue
        if pid not in interp.pid_models:
            print(f"skip '{sp}': not in the model"); continue
        print(f"== {sp} ==")
        run_species(interp, sampler, library, pid, args.n_sample, args.seed,
                    args.outdir, args.depth_res_cm)


if __name__ == "__main__":
    main()
