"""
diagnostics.py -- evaluate the modeling strategy behind shower_gamma_model.

Plot sets, each answering a different question:
  fit      : is a gamma-mixture the right representation? (fit overlays + residuals)
  decomp   : are m and the (mode,width) clouds sensible / Gaussian-shaped?
  sampfid  : does the sampler reproduce the fitted distributions? (fits vs samples)
  loo      : is the energy interpolation reliable? (leave-one-energy-out)
  subcasc  : distribution of the G4 sub-cascade count, at all 5 thresholds
             (the threshold-calibration plot)

Inputs:
  --dists   shower_model_dists.pkl   (raw per-energy fit summaries)
  --model   shower_model.pkl         (the sampler; needed for 'sampfid')
  --g4-dir  ../output                (the .h5 library; needed for 'fit' and 'subcasc')

Example:
  python diagnostics.py --dists ../output/shower_model_dists.pkl \
      --model ../output/shower_model.pkl --g4-dir ../output \
      --species pip KS --energies 100 1000 10000
"""
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shower_gamma_model as M   # fit_profile, _kernel, _from_z, load_g4_library, load_model, classes

# ---------------------------------------------------------------------------
# Global style: big fonts + LaTeX-style (mathtext) + scientific-notation axes.
# NOTE: we use matplotlib's built-in mathtext ($...$), which renders LaTeX-style
# math WITHOUT needing a system LaTeX install. If you have LaTeX and want the
# real thing, set "text.usetex": True below.
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.size":              20,
    "axes.titlesize":         20,
    "axes.labelsize":         20,
    "xtick.labelsize":        15,
    "ytick.labelsize":        15,
    "legend.fontsize":        14,
    "figure.titlesize":       22,
    "mathtext.fontset":       "cm",       # Computer-Modern-like math
    "axes.formatter.use_mathtext": True,  # 1e6 -> x10^6 on axes
    "axes.formatter.limits":  (-3, 4),    # switch to sci notation outside 1e-3..1e4
})

# LaTeX species labels and a scientific-notation energy formatter.
LATEX = {211: r"$\pi^+$", -211: r"$\pi^-$", 111: r"$\pi^0$",
         321: r"$K^+$", -321: r"$K^-$", 310: r"$K_S$", 130: r"$K_L$",
         2212: r"$p$", 2112: r"$n$"}


def _lab(pid):
    return LATEX.get(pid, M.PID_TO_NAME.get(pid, str(pid)))


def _sci(E):
    """Energy as a LaTeX power of ten, e.g. 1000 -> '10^{3}'."""
    e = np.log10(E)
    if abs(e - round(e)) < 1e-6:
        return rf"10^{{{int(round(e))}}}"
    mant, powr = f"{E:.0e}".split("e")
    return rf"{mant}\times 10^{{{int(powr)}}}"


# ---------------------------------------------------------------------------
# Which diagnostics to make (flip to False to skip). --plots overrides.
# ---------------------------------------------------------------------------
PLOT_FIT     = True    # fit-quality overlays + residual histograms  (needs --g4-dir)
PLOT_DECOMP  = True    # p(m) vs E, and (mode,width) clouds per m
PLOT_SAMPFID = True    # fits vs samples for alpha & beta            (needs --model)
PLOT_LOO     = True    # leave-one-energy-out interpolation error
PLOT_SUBCASC = True    # G4 sub-cascade multiplicity, all thresholds (needs --g4-dir)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _mode_width(alpha, beta):
    """Physical shape of a gamma component: peak depth and rms width [cm]."""
    mode = np.maximum(alpha - 1.0, 1e-6) / beta
    width = np.sqrt(alpha) / beta
    return mode, width


def _reconstruct(fit, x):
    return fit["N"] * sum(fit["w"][c] * M._kernel(x, fit["alpha"][c], fit["beta"][c])
                          for c in range(fit["m"]))


# ===========================================================================
# 1. FIT QUALITY  (needs the .h5 profiles; refits a subset on the fly)
# ===========================================================================
def plot_fit_quality(g4_dir, pid, E, out, n_show=12, n_resid=400, seed=0, Kmax=M.KMAX_DEFAULT):
    lib = M.load_g4_library(g4_dir)
    if pid not in lib or E not in lib[pid]:
        print(f"[fit]  no data for pid={pid} E={E}"); return
    x = lib[pid][E]["z_centers"]; profs = lib[pid][E]["profiles"]
    name = M.PID_TO_NAME.get(pid, str(pid)); lab = _lab(pid)
    rng = np.random.default_rng(seed)

    # (a) overlay grid of random runs: G4 + fit + components
    idx = rng.choice(len(profs), size=min(n_show, len(profs)), replace=False)
    ncol = 4; nrow = int(np.ceil(len(idx) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.4 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, i in zip(ax, idx):
        y = profs[i]; fit = M.fit_profile(x, y, Kmax)
        a.plot(x, y, color="0.6", lw=1.2, label="G4")
        a.plot(x, _reconstruct(fit, x), "r", lw=1.8, label=f"fit $m={fit['m']}$")
        for c in range(fit["m"]):
            a.plot(x, fit["N"] * fit["w"][c] * M._kernel(x, fit["alpha"][c], fit["beta"][c]),
                   "b--", lw=1.0)
        a.set_title(f"run {i},  $m={fit['m']}$", fontsize=14); a.set_xlim(0, None)
        a.set_xlabel(r"depth [cm]", fontsize=14)
    for a in ax[len(idx):]:
        a.set_visible(False)
    ax[0].legend(fontsize=13)
    fig.suptitle(rf"Fit overlays — {lab} at $E={_sci(E)}$ GeV  ({len(idx)} random runs)")
    fig.tight_layout()
    p = os.path.join(out, f"diag_fitgrid_{name}_E{int(E)}.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)

    # (b) residual histogram
    ridx = rng.choice(len(profs), size=min(n_resid, len(profs)), replace=False)
    rr = []
    for i in ridx:
        y = profs[i]; fit = M.fit_profile(x, y, Kmax)
        rr.append(np.sqrt(np.mean((y - _reconstruct(fit, x)) ** 2)) / max(y.max(), 1e-9))
    rr = np.array(rr)
    fig, a = plt.subplots(figsize=(8, 6))
    a.hist(rr, 40, color="steelblue")
    a.set_xlabel(r"relative RMS residual ($/$ peak)"); a.set_ylabel("runs")
    a.set_title(rf"Fit residuals — {lab}, $E={_sci(E)}$ GeV" "\n"
                rf"median $= {np.median(rr):.3f}$")
    fig.tight_layout()
    p = os.path.join(out, f"diag_fitresid_{name}_E{int(E)}.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)


# ===========================================================================
# 2. DECOMPOSITION  (from dists)
# ===========================================================================
def plot_decomposition(dists, pid, energies, out, Kmax=M.KMAX_DEFAULT):
    name = M.PID_TO_NAME.get(pid, str(pid)); lab = _lab(pid)
    ed = dists.get(pid, {})
    if not ed:
        print(f"[decomp] no dists for {name}"); return
    all_E = sorted(ed.keys())

    # p(m) vs energy
    fig, a = plt.subplots(figsize=(8, 6))
    for k in range(Kmax):
        a.plot(all_E, [ed[E]["p_m"][k] for E in all_E], "o-", lw=2, ms=7, label=rf"$m={k+1}$")
    a.set_xscale("log"); a.set_xlabel(r"$E$ [GeV]"); a.set_ylabel(r"$p(m)$")
    a.legend(); a.set_title(rf"$p(m\,|\,E)$ — {lab}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_pm_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # per m: (mode,width) scatter, one subpanel per requested energy
    for m in range(1, Kmax + 1):
        Es = [E for E in energies if E in ed and m in ed[E]["Z"]]
        if not Es:
            continue
        fig, ax = plt.subplots(1, len(Es), figsize=(5.2 * len(Es), 4.6), squeeze=False)
        ax = ax[0]
        for a, E in zip(ax, Es):
            Z = ed[E]["Z"][m]
            modes, widths = [], []
            for row in Z:
                _, al, be = M._from_z(row, m)
                mo, wi = _mode_width(al, be)
                modes.append(mo); widths.append(wi)
            modes = np.concatenate(modes); widths = np.concatenate(widths)
            a.scatter(modes, widths, s=10, alpha=0.3, color="purple")
            a.set_title(rf"$E={_sci(E)}$ GeV,  $m={m}$  ($n={len(Z)}$)", fontsize=16)
            a.set_xlabel(r"mode [cm]"); a.set_ylabel(r"width [cm]")
        fig.suptitle(rf"Component (mode, width) — {lab},  $m={m}$")
        fig.tight_layout()
        p = os.path.join(out, f"diag_decomp_{name}_m{m}.png"); fig.savefig(p, dpi=140); plt.close(fig)
        print("saved", p)


# ===========================================================================
# 3. SAMPLING FIDELITY  (dists = fitted side, model = sampled side)
# ===========================================================================
def _fitted_vals(dists, pid, E, m, which):
    Z = dists[pid][E]["Z"].get(m)
    if Z is None:
        return np.array([])
    out = []
    for row in Z:
        _, al, be = M._from_z(row, m)
        out.append(al if which == "alpha" else be)
    return np.concatenate(out) if out else np.array([])


def _sampled_vals(sampler, pid, E, m, which, n_target=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    tries = 0
    while sum(len(v) for v in out) < n_target and tries < n_target * 30:
        tries += 1
        mm, w, al, be = sampler.sample_params(pid, E, rng)
        if mm != m:
            continue
        out.append(al if which == "alpha" else be)
    return np.concatenate(out) if out else np.array([])


def plot_sampling_fidelity(dists, model, pid, energies, out, Kmax=M.KMAX_DEFAULT):
    name = M.PID_TO_NAME.get(pid, str(pid)); lab = _lab(pid)
    sampler = M.ShowerSampler(model)
    Es = [E for E in energies if E in dists.get(pid, {})]
    ms = list(range(1, Kmax + 1))
    sym = {"alpha": r"\alpha", "beta": r"\beta"}
    for which in ("alpha", "beta"):
        fig, ax = plt.subplots(len(Es), len(ms), figsize=(4.6 * len(ms), 4.0 * len(Es)),
                               squeeze=False)
        rng = np.random.default_rng(1)
        for r, E in enumerate(Es):
            for c, m in enumerate(ms):
                a = ax[r][c]
                fv = _fitted_vals(dists, pid, E, m, which)
                sv = _sampled_vals(sampler, pid, E, m, which, rng=rng)
                if len(fv) == 0 and len(sv) == 0:
                    a.set_visible(False); continue
                lo = min([v.min() for v in (fv, sv) if len(v)])
                hi = max([v.max() for v in (fv, sv) if len(v)])
                bins = np.linspace(lo, hi, 40)
                if len(fv):
                    a.hist(fv, bins, density=True, alpha=0.5, color="k", label="fits")
                if len(sv):
                    a.hist(sv, bins, density=True, alpha=0.5, color="r", label="samples")
                a.set_title(rf"$E={_sci(E)}$ GeV,  $m={m}$", fontsize=15)
                a.set_xlabel(rf"${sym[which]}$", fontsize=17)
                if r == 0 and c == 0:
                    a.legend(fontsize=13)
        fig.suptitle(rf"Fits vs samples: ${sym[which]}$ — {lab}  (should overlap)")
        fig.tight_layout()
        p = os.path.join(out, f"diag_sampfid_{which}_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig)
        print("saved", p)


# ===========================================================================
# 4. LEAVE-ONE-ENERGY-OUT interpolation check  (from dists)
# ===========================================================================
def plot_loo(dists, pid, out, Kmax=M.KMAX_DEFAULT):
    name = M.PID_TO_NAME.get(pid, str(pid)); lab = _lab(pid)
    ed = dists.get(pid, {})
    Es = sorted(ed.keys())
    if len(Es) < 4:
        print(f"[loo] need >=4 energies for {name}"); return
    held, rel_pm = [], []
    for Eh in Es:
        sub = {pid: {E: ed[E] for E in Es if E != Eh}}
        it = M.ShowerParamInterpolator(sub, Kmax=Kmax)
        pred = it.p_m(pid, Eh); true = ed[Eh]["p_m"]
        held.append(Eh)
        rel_pm.append(np.abs(pred - true) / np.maximum(true, 1e-3))
    held = np.array(held); rel_pm = np.array(rel_pm)
    fig, a = plt.subplots(figsize=(9, 5.5))
    for k in range(Kmax):
        a.plot(held, rel_pm[:, k], "o-", lw=2, ms=7, label=rf"$p(m={k+1})$")
    a.axhline(0.05, color="green", ls="--", lw=1.2, label=r"$5\%$ ref")
    a.set_xscale("log"); a.set_xlabel(r"held-out $E$ [GeV]")
    a.set_ylabel("relative interpolation error"); a.legend()
    a.set_title(rf"Leave-one-energy-out $p(m)$ — {lab}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_loo_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


# ===========================================================================
# 5. SUB-CASCADE MULTIPLICITY  (from the G4 .h5; all thresholds overlaid)
# ===========================================================================
def plot_subcascade_dist(g4_dir, pid, energies, out):
    import h5py
    name = M.PID_TO_NAME.get(pid, str(pid)); lab = _lab(pid)
    Es = list(energies)
    fig, ax = plt.subplots(1, len(Es), figsize=(6.4 * len(Es), 5.4), squeeze=False)
    ax = ax[0]
    cmap = plt.get_cmap("viridis")
    any_data = False
    for a, E in zip(ax, Es):
        fname = os.path.join(g4_dir, f"shower_{name}_E{int(E)}GeV.h5")
        if not os.path.exists(fname):
            a.set_title(rf"$E={_sci(E)}$ GeV" "\n(no file)", fontsize=15); continue
        with h5py.File(fname, "r") as f:
            if "n_subcascades" not in f:
                a.set_title(rf"$E={_sci(E)}$ GeV" "\n(no n_subcascades)", fontsize=15); continue
            nsc = f["n_subcascades"][:]
            thr = np.asarray(f.attrs.get("subcascade_thresholds",
                                         [0.01, 0.02, 0.05, 0.10, 0.20]))
        any_data = True
        nt = nsc.shape[1]
        mx = int(nsc.max()) if nsc.size else 1
        bins = np.arange(-0.5, mx + 1.5, 1.0)
        for j in range(nt):
            a.hist(nsc[:, j], bins=bins, histtype="step", lw=2.4, density=True,
                   color=cmap(j / max(nt - 1, 1)),
                   label=rf"$>{thr[j]*100:.0f}\%\ E_{{\rm prim}}$")
        a.set_title(rf"$E={_sci(E)}$ GeV", fontsize=17)
        a.set_xlabel(r"$N_{\rm sub\ cascades}$")
        a.set_ylabel("fraction of showers")
        a.legend(title="threshold", fontsize=12, title_fontsize=13)
    fig.suptitle(rf"Sub-cascade multiplicity — {lab}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_subcasc_{name}.png")
    if any_data:
        fig.savefig(p, dpi=140); print("saved", p)
    else:
        print(f"[subcasc] no n_subcascades data for {name} at {Es}")
    plt.close(fig)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--g4-dir", default=None)
    ap.add_argument("--species", nargs="+", default=["pip"])
    ap.add_argument("--energies", nargs="+", type=float, default=[1000.0, 10000.0])
    ap.add_argument("--out", default=None,
                    help="output folder (default: a 'diagnostics' folder next to --dists)")
    ap.add_argument("--plots", nargs="+", default=None,
                    choices=["all", "fit", "decomp", "sampfid", "loo", "subcasc"],
                    help="override the PLOT_* flags at the top of the file")
    args = ap.parse_args()

    want = {n for n, on in (("fit", PLOT_FIT), ("decomp", PLOT_DECOMP),
                            ("sampfid", PLOT_SAMPFID), ("loo", PLOT_LOO),
                            ("subcasc", PLOT_SUBCASC)) if on}
    if args.plots:
        want = ({"fit", "decomp", "sampfid", "loo", "subcasc"}
                if "all" in args.plots else set(args.plots))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.dists)), "diagnostics")
    os.makedirs(out, exist_ok=True)

    dists = _load_pickle(args.dists)
    model = M.load_model(args.model) if args.model else None

    for sp in args.species:
        pid = M.NAME_TO_PID[sp]
        if "decomp" in want:
            plot_decomposition(dists, pid, args.energies, out)
        if "loo" in want:
            plot_loo(dists, pid, out)
        if "sampfid" in want:
            if model is None:
                print("[sampfid] skipped: pass --model")
            else:
                plot_sampling_fidelity(dists, model, pid, args.energies, out)
        if "subcasc" in want:
            if not args.g4_dir:
                print("[subcasc] skipped: pass --g4-dir")
            else:
                plot_subcascade_dist(args.g4_dir, pid, args.energies, out)
        if "fit" in want:
            if not args.g4_dir:
                print("[fit] skipped: pass --g4-dir")
            else:
                for E in args.energies:
                    plot_fit_quality(args.g4_dir, pid, E, out)
    print("done ->", out)


if __name__ == "__main__":
    main()
