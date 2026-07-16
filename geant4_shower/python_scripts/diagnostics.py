"""
diagnostics.py -- evaluate the modeling strategy behind shower_gamma_model.

Four plot sets, each answering a different question:
  fit     : is a gamma-mixture the right representation? (fit overlays + residuals)
  decomp  : are m and the (mode,width) clouds sensible / Gaussian-shaped?
  sampfid : does the sampler reproduce the fitted distributions? (fits vs samples)
  loo     : is the energy interpolation reliable? (leave-one-energy-out)

Inputs:
  --dists   shower_model_dists.pkl   (raw per-energy fit summaries)
  --model   shower_model.pkl         (the sampler; needed for 'sampfid')
  --g4-dir  ../output                (the .h5 library; needed for 'fit')

Example:
  python diagnostics.py --dists ../output/shower_model_dists.pkl \
      --model ../output/shower_model.pkl --g4-dir ../output \
      --species pip p --energies 1000 10000 --out ../output/diag --plots all
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
# Choose which diagnostics to make (flip to False to skip).
# The CLI flag --plots overrides these if given.
# ---------------------------------------------------------------------------
PLOT_FIT     = True    # fit-quality overlays + residual histograms  (needs --g4-dir)
PLOT_DECOMP  = True    # p(m) vs E, and (mode,width) clouds per m
PLOT_SAMPFID = True    # fits vs samples for alpha & beta            (needs --model)
PLOT_LOO     = True    # leave-one-energy-out interpolation error


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
    name = M.PID_TO_NAME.get(pid, str(pid))
    rng = np.random.default_rng(seed)

    # (a) overlay grid of 12 random runs: G4 + fit + components
    idx = rng.choice(len(profs), size=min(n_show, len(profs)), replace=False)
    ncol = 4; nrow = int(np.ceil(len(idx) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
    ax = np.atleast_1d(ax).ravel()
    for a, i in zip(ax, idx):
        y = profs[i]; fit = M.fit_profile(x, y, Kmax)
        a.plot(x, y, color="0.6", lw=1.0, label="G4")
        a.plot(x, _reconstruct(fit, x), "r", lw=1.4, label=f"fit m={fit['m']}")
        for c in range(fit["m"]):
            a.plot(x, fit["N"] * fit["w"][c] * M._kernel(x, fit["alpha"][c], fit["beta"][c]),
                   "b--", lw=0.7)
        a.set_title(f"run {i}  m={fit['m']}", fontsize=8); a.set_xlim(0, None)
    for a in ax[len(idx):]:
        a.set_visible(False)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Fit overlays -- {name} at {E:.0f} GeV  (12 random runs)")
    fig.tight_layout()
    p = os.path.join(out, f"diag_fitgrid_{name}_E{int(E)}.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)

    # (b) residual histogram over n_resid random runs
    ridx = rng.choice(len(profs), size=min(n_resid, len(profs)), replace=False)
    rr = []
    for i in ridx:
        y = profs[i]; fit = M.fit_profile(x, y, Kmax)
        rr.append(np.sqrt(np.mean((y - _reconstruct(fit, x)) ** 2)) / max(y.max(), 1e-9))
    rr = np.array(rr)
    fig, a = plt.subplots(figsize=(6, 4))
    a.hist(rr, 40, color="steelblue")
    a.set_xlabel("relative RMS residual (/ peak)"); a.set_ylabel("runs")
    a.set_title(f"Fit residuals -- {name} at {E:.0f} GeV   median={np.median(rr):.3f}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_fitresid_{name}_E{int(E)}.png"); fig.savefig(p, dpi=140); plt.close(fig)
    print("saved", p)


# ===========================================================================
# 2. DECOMPOSITION  (from dists)
# ===========================================================================
def plot_decomposition(dists, pid, energies, out, Kmax=M.KMAX_DEFAULT):
    name = M.PID_TO_NAME.get(pid, str(pid))
    ed = dists.get(pid, {})
    if not ed:
        print(f"[decomp] no dists for {name}"); return
    all_E = sorted(ed.keys())

    # p(m) vs energy
    fig, a = plt.subplots(figsize=(6, 4))
    for k in range(Kmax):
        a.plot(all_E, [ed[E]["p_m"][k] for E in all_E], "o-", label=f"m={k+1}")
    a.set_xscale("log"); a.set_xlabel("E [GeV]"); a.set_ylabel("p(m)")
    a.legend(); a.set_title(f"p(m | E) -- {name}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_pm_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)

    # per m: (mode,width) scatter, one subpanel per requested energy
    for m in range(1, Kmax + 1):
        Es = [E for E in energies if E in ed and m in ed[E]["Z"]]
        if not Es:
            continue
        fig, ax = plt.subplots(1, len(Es), figsize=(4 * len(Es), 3.6), squeeze=False)
        ax = ax[0]
        for a, E in zip(ax, Es):
            Z = ed[E]["Z"][m]
            modes, widths = [], []
            for row in Z:
                _, al, be = M._from_z(row, m)
                mo, wi = _mode_width(al, be)
                modes.append(mo); widths.append(wi)
            modes = np.concatenate(modes); widths = np.concatenate(widths)
            a.scatter(modes, widths, s=5, alpha=0.3, color="purple")
            a.set_title(f"{name}  E={E:.0f}  m={m}  (n={len(Z)})", fontsize=9)
            a.set_xlabel("mode [cm]"); a.set_ylabel("width [cm]")
        fig.suptitle(f"component (mode, width) clouds -- {name}, m={m}")
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
    name = M.PID_TO_NAME.get(pid, str(pid))
    sampler = M.ShowerSampler(model)
    Es = [E for E in energies if E in dists.get(pid, {})]
    ms = list(range(1, Kmax + 1))
    for which in ("alpha", "beta"):
        fig, ax = plt.subplots(len(Es), len(ms), figsize=(3.4 * len(ms), 3 * len(Es)),
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
                a.set_title(f"E={E:.0f}  m={m}", fontsize=8)
                if r == 0 and c == 0:
                    a.legend(fontsize=7)
        fig.suptitle(f"fits vs samples: {which} -- {name}  (should overlap)")
        fig.tight_layout()
        p = os.path.join(out, f"diag_sampfid_{which}_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig)
        print("saved", p)


# ===========================================================================
# 4. LEAVE-ONE-ENERGY-OUT interpolation check  (from dists)
# ===========================================================================
def plot_loo(dists, pid, out, Kmax=M.KMAX_DEFAULT):
    name = M.PID_TO_NAME.get(pid, str(pid))
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
    fig, a = plt.subplots(figsize=(7, 4))
    for k in range(Kmax):
        a.plot(held, rel_pm[:, k], "o-", label=f"p(m={k+1})")
    a.axhline(0.05, color="green", ls="--", lw=0.8, label="5% ref")
    a.set_xscale("log"); a.set_xlabel("held-out E [GeV]")
    a.set_ylabel("relative interpolation error"); a.legend()
    a.set_title(f"leave-one-energy-out (p(m)) -- {name}")
    fig.tight_layout()
    p = os.path.join(out, f"diag_loo_{name}.png"); fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


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
                    choices=["all", "fit", "decomp", "sampfid", "loo"],
                    help="override the PLOT_* flags at the top of the file")
    args = ap.parse_args()

    # which plots: PLOT_* flags by default; --plots overrides if given
    want = {n for n, on in (("fit", PLOT_FIT), ("decomp", PLOT_DECOMP),
                            ("sampfid", PLOT_SAMPFID), ("loo", PLOT_LOO)) if on}
    if args.plots:
        want = {"fit", "decomp", "sampfid", "loo"} if "all" in args.plots else set(args.plots)

    # every plot lands in a 'diagnostics' folder inside output/ (next to the pkl)
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
        if "fit" in want:
            if not args.g4_dir:
                print("[fit] skipped: pass --g4-dir")
            else:
                for E in args.energies:
                    plot_fit_quality(args.g4_dir, pid, E, out)
    print("done ->", out)


if __name__ == "__main__":
    main()
