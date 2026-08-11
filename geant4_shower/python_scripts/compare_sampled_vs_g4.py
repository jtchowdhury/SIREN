"""
compare_sampled_vs_g4.py
========================
Performance comparison of our shower model to Geant4 showers using L2 and KS metrics.

Here we compare a shower our model GENERATED to a shower Geant4 produced:
  - draw a shower from the G4 data sample,
  - draw a shower from our sampled model (gamma mixture / single gamma / analytic),
  - pair them (random pairing -- they have no 1-to-1 correspondence),
  - score the pair with a plain L2 and a KS, and average over many pairs.
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
import compare_fixed2gamma as F   # its build_ensembles also yields the fixed 2-gamma

# same colours as perevent_l2ks; labels reflect that these are SAMPLES, not fits
SAMPLED_META = {
    "analytic": ("Analytic (fixed shape)", "#4682B4"),   # steel blue
    "single":   ("Sampled Single Gamma",   "#BA55D3"),   # medium orchid
    "twogamma": ("Sampled 2-Gamma",         "lightseagreen"),
    "mixture":  ("Gamma Mixture Model",     "#FA8072"),   # salmon
}


def sampled_vs_g4(g4, gen, x, sigma_cm, rng):
    """Randomly pair generated showers with (blurred) G4 showers; return the
    mean per-pair relative L2 and mean KS (normalized-profile CDF gap)."""
    binw = float(x[1] - x[0])
    do_blur = sigma_cm > 0                       # sigma_cm<=0 -> compare RAW G4
    sig_bins = max(sigma_cm / binw, 1e-6)
    ng, ns = len(g4), len(gen)
    n = min(ng, ns)
    gi = rng.permutation(ng)[:n]
    si = rng.permutation(ns)[:n]
    l2s, kss = [], []
    for a, b in zip(gi, si):
        ga = np.asarray(g4[a], float)
        g = gaussian_filter1d(ga, sig_bins, mode="constant") if do_blur else ga
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


def g4_vs_g4(g4, x, sigma_cm, rng):
    """Empirical floor: compare one (blurred) G4 shower to ANOTHER raw G4 shower,
    scored exactly like the model comparison (disjoint halves, no self-pairing)."""
    m = len(g4)
    half = m // 2
    if half < 1:
        return np.nan, np.nan
    perm = rng.permutation(m)
    A = g4[perm[:half]]
    B = g4[perm[half:2 * half]]
    return sampled_vs_g4(A, B, x, sigma_cm, rng)


def run_species(interp, sampler, library, pid, n_sample, seed, outdir, sigma_cm, suffix=""):
    name = PID_TO_NAME.get(pid, str(pid))
    rng = np.random.default_rng(seed)
    rows = []
    for E in sorted(library[pid].keys()):
        x = library[pid][E]["z_centers"]
        ens = F.build_ensembles(interp, sampler, library, pid, E, x, n_sample,
                                rng, fix_yield=True)   # G4, M1, SG, F2, M3
        g4 = ens["G4"]
        for tag, form in (("M1", "analytic"), ("SG", "single"),
                          ("F2", "twogamma"), ("M3", "mixture")):
            gen = ens[tag]
            if gen is None:
                continue
            l2, ks = sampled_vs_g4(g4, gen, x, sigma_cm, rng)
            rows.append(dict(species=name, E=E, form=form, l2=l2, ks=ks))

        # empirical floor: one G4 shower vs another G4 shower
        fl2, fks = g4_vs_g4(g4, x, sigma_cm, rng)
        rows.append(dict(species=name, E=E, form="g4floor", l2=fl2, ks=fks))
        # theoretical two-sample KS floor (95% critical, n=m=N_G4)
        N = len(g4)
        rows.append(dict(species=name, E=E, form="ks_theory", l2=float("nan"),
                         ks=float(1.358 * np.sqrt(2.0 / N)) if N > 0 else float("nan")))

        msg = "  ".join(f"{r['form']}:KS={r['ks']:.2g}" for r in rows if r["E"] == E
                        and r["form"] in ("single", "twogamma", "mixture", "g4floor"))
        print(f"  E={E:8.0f}  {msg}")

    C._write_l2ks_csv(rows, os.path.join(outdir, f"sampled_vs_g4_{name}{suffix}.csv"))
    _plot(rows, name, outdir, sigma_cm, suffix)
    return rows


def _plot(rows, name, outdir, sigma_cm, suffix=""):
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
        Ef, vf = C._series(rows, "g4floor", val_key, match_key="form")
        if len(Ef):
            ax.plot(Ef, vf, "--", color="#444444", lw=2.0, alpha=0.9,
                    label="G4 vs G4 (empirical floor)")
        '''
        if val_key == "ks":
            Et, vt = C._series(rows, "ks_theory", "ks", match_key="form")
            if len(Et):
                ax.plot(Et, vt, ":", color="#c0392b", lw=2.4, alpha=0.95,
                        label=r"Theory floor (95% KS)")
        '''
        ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=12, length=6)

    draw(ax1, "l2")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$L_2$ ($\sum(\mathrm{data}-\mathrm{model})^2/\sum\mathrm{data}^2$)",
                   fontsize=13)
    tnote = "" if sigma_cm > 0 else "  [unblurred G4]"
    ax1.set_title(f"Sampled Model Shower vs Geant4 Shower ({sp}){tnote}\n",
                  fontsize=15, fontweight="bold", pad=10)
    ax1.legend(fontsize=11, framealpha=0.92, loc="best")

    draw(ax2, "ks")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylabel("KS statistic", fontsize=14)
    ax2.set_xlabel("Shower Energy [GeV]", fontsize=15)
    ax2.legend(fontsize=11, framealpha=0.92, loc="best")

    fig.tight_layout()
    out = os.path.join(outdir, f"sampled_vs_g4_{name}{suffix}.png")
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
    ap.add_argument("--no-blur", action="store_true",
                    help="compare against RAW (unblurred) G4; disables the ~45cm detector blur. "
                         "Writes *_noblur.png/.csv so the blurred results aren't overwritten.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/result")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_path = args.model or os.path.join(args.g4_dir, "shower_model.pkl")
    sigma_cm = 0.0 if args.no_blur else args.depth_res_cm
    suffix = "_noblur" if args.no_blur else ""

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
                    args.outdir, sigma_cm, suffix)


if __name__ == "__main__":
    main()
