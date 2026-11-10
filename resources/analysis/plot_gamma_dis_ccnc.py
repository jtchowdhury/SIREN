#!/usr/bin/env python3
"""
plot_gamma_dis_ccnc.py -- whole-event Cherenkov showers, CC vs NC.

Inputs: the samples written by gamma_dis_ccnc_samples.py.

F1  spaghetti at fixed E_nu   (ccnc_mono_E<E>_{cc,nc}.npz)
    100 CC and 100 NC event showers, each built ONLY from SIREN's output
    subshower parameters  f(x) = Sum_k amp_k Gamma(x; alpha_k, beta_k),
    normalised to unit area (same axes / style as the e- vs pi+ figure).
    CC includes the electron subshower (parent -1); NC has no charged lepton.
F2  spaghetti at fixed E_had  (events selected from ccnc_range_{cc,nc}.npz with
    E_had within +-tol of the target). Separates "CC has an electron" from
    "NC has less energy".
F3  L2 / KS vs E_had (and vs E_nu), one figure per channel.
    For every SIREN event (= one boosted PYTHIA hadron list + the lepton):
      * draw --n-g4 (100) G4 event showers: one single-particle G4 shower per
        hadron (at its kinetic energy), plus one for the electron in CC;
      * GammaDISFromSpline: the event's output subshowers (all, incl. the
        electron), compared with each of the 100 G4 showers -> mean L2, mean KS;
      * Analytic (DISFromSpline): one analytic shower of the hadronic blob only
        (no electron), compared with each of the same 100 G4 showers;
      * floor: one extra G4 shower vs the same 100.
    The per-event means are then averaged in energy bins. Metric and depth blur
    identical to compare_event_vs_g4.py.

Analytic formula (profile in depth x [cm], t = x / X0):
  hadronic blob:  Aachen pi+ profile (Raedel & Wiebusch Geant4 parametrization for
                  ice, as implemented in fennel): Gamma(t; a, b), t = x / X0,
                  a = 1.81098 + 0.90572 log10(E/GeV), b = 0.34347, X0 = 39.30 cm,
                  E = E_had - nucleon mass (kinetic), scaled to the model's mean
                  pi+ yield at E. KS does not depend on that scale; L2 does.

Usage:
    python plot_gamma_dis_ccnc.py --samples output/gamma_dis_samples \
        --g4-dir DIR [DIR ...] --outdir output/gamma_dis_plots/ccnc
    (several --g4-dir folders are merged; without --g4-dir only F1 and F2 are made)
"""
import os
import sys
import csv
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gamma_dis_common as G
from plot_gamma_dis_showers import binned                       # also puts SIREN/python on sys.path
import hadronic_showers as hs                                   # noqa: E402

COL = {"cc": "#4c78a8", "nc": "#f58518"}
CURVES = {"gamma":    ("GammaDISFromSpline vs G4", "#F08070", "o", "-"),
          "analytic": ("Analytic (DISFromSpline blob) vs G4", "#4682B4", "s", "-"),
          "floor":    ("G4 vs G4 (empirical floor)", "#444444", None, "--")}
CURVE_ORDER = ("gamma", "analytic", "floor")    # column order in the rows

# analytic constants (see docstring)
AACHEN_PIP_ALPHA, AACHEN_PIP_BETA, AACHEN_PIP_B = 1.81098, 0.90572, 0.34347
AACHEN_X0_CM = 36.08 / 0.9180      # fennel: X0 = 36.08 g/cm^2, ice density 0.918 g/cm^3
M_NUCLEON = 0.5 * (G.REST_MASS[2212] + G.REST_MASS[2112])   # isoscalar target


# --------------------------------------------------------------------------
#  profiles
# --------------------------------------------------------------------------
def output_profile(s, e, x, hadrons_only=False):
    """Event shower from SIREN's output subshowers (all parents, or only the
    hadrons, parent >= 0, i.e. without the electron)."""
    a, b = s["sub_off"][e], s["sub_off"][e + 1]
    y = np.zeros_like(x)
    for k in range(a, b):
        if hadrons_only and s["sub_parent"][k] < 0:
            continue
        y += s["sub_amp"][k] * hs._kernel(x, s["sub_alpha"][k], s["sub_beta"][k])
    return y


def analytic_had(x, E, yield_tot):
    """Aachen (Raedel & Wiebusch, Geant4 in ice) pi+ longitudinal profile, as
    implemented in fennel (fennel/data/aachen.pkl, 'hadron cascade' ->
    'longitudinal parameters' -> 211):  Gamma(t; a, b), t = x / X0,
    a = 1.81098 + 0.90572 log10(E/GeV), b = 0.34347. Unit area in cm, times yield."""
    a = max(AACHEN_PIP_ALPHA + AACHEN_PIP_BETA * np.log10(E), 1e-3)
    return yield_tot * hs._kernel(x / AACHEN_X0_CM, a, AACHEN_PIP_B) / AACHEN_X0_CM


def mean_yield(model, species, E):
    """Model mean yield with the SAME out-of-range rule as model.sample():
    clamp to the trained energy range, then scale linearly (yield ~ E).
    model.yield_mean alone stays frozen above the range (e.g. e- is trained only
    to 10 TeV), which would starve the analytic of light."""
    lo, hi = model.energy_range(species)
    Ec = min(max(E, lo), hi)
    return model.yield_mean(species, Ec) * E / Ec


def unit_area(x, y):
    area = hs._trapz(y, x)
    return y / area if area > 0 else y


# --------------------------------------------------------------------------
#  F1 / F2: spaghetti
# --------------------------------------------------------------------------
def spaghetti(groups, title, path, xlim):
    """groups: list of (channel, sample, event indices, label)."""
    x = np.linspace(0.5, xlim[1] + 200, 2 * int(xlim[1] + 200))
    fig, ax = plt.subplots(figsize=(8, 6))
    for ch, s, idx, lab in groups:
        for i, e in enumerate(idx):
            ax.plot(x, unit_area(x, output_profile(s, e, x)), color=COL[ch], lw=0.5,
                    alpha=0.5, label=lab if i == 0 else None)
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.legend()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    hs._finish(fig, ax, save=path, show=False)
    plt.close(fig)


def pick(rng, idx, n):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


# --------------------------------------------------------------------------
#  F3: L2 / KS vs G4
# --------------------------------------------------------------------------
def load_g4_tools(g4_dirs):
    sys.path.insert(0, os.path.join(G.SIREN_ROOT, "geant4_shower", "python_scripts"))
    import compare_event_vs_g4 as EV                               # noqa: E402
    import compare_methods as C                                    # noqa: E402
    from shower_gamma_model import load_g4_library                 # noqa: E402
    lib = {}
    for d in g4_dirs:
        if not glob.glob(os.path.join(d, "shower_*_E*GeV.h5")):
            raise SystemExit(f"no shower_*_E*GeV.h5 files in {d}")
        for pid, byE in load_g4_library(d).items():       # merge folders
            lib.setdefault(pid, {}).update(byE)
    return EV, C, lib


def blur(G4, sigma, binw):
    """Detector depth blur of G4 profiles (rows), exactly as compare_event_vs_g4._l2_ks."""
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return np.asarray(G4, float)
    return gaussian_filter1d(np.asarray(G4, float), max(sigma / binw, 1e-6),
                             axis=-1, mode="constant")


def l2_ks_many(Gb, test):
    """L2 and KS of one test profile against many (already blurred) G4 profiles.
    Same formulas as compare_event_vs_g4._l2_ks, vectorised over the rows of Gb."""
    s = np.asarray(test, float)
    denom = np.sum(Gb * Gb, axis=1)
    l2 = np.where(denom > 0, np.sum((Gb - s) ** 2, axis=1) / np.where(denom > 0, denom, 1), np.nan)
    gp, sp = np.clip(Gb, 0, None), np.clip(s, 0, None)
    gs, ss = gp.sum(axis=1), sp.sum()
    if ss <= 0:
        return l2, np.full(len(Gb), np.nan)
    cg = np.cumsum(gp, axis=1) / np.where(gs > 0, gs, 1)[:, None]
    ks = np.where(gs > 0, np.max(np.abs(cg - np.cumsum(sp) / ss), axis=1), np.nan)
    return l2, ks


def g4_composite(EV, lib, parts, x_len, rng):
    """One G4 event shower: sum of one single-particle G4 shower per hadron."""
    tot = np.zeros(x_len)
    for pid, E in parts:
        tot += EV._g4_draw(lib, pid, E, rng)
    return tot


def score_channel(EV, lib, model, s, ch, x, sigma, binw, rng, max_events, max_missing, n_g4):
    n_ev = len(s["E_nu"])
    events = pick(rng, np.arange(n_ev), max_events)
    rows, n_skip, missing_pids = [], 0, {}
    for j, e in enumerate(events):
        h0, h1 = s["had_off"][e], s["had_off"][e + 1]
        # hadrons at kinetic energy (the G4 library / model are labelled by E_kin)
        parts = [(int(s["had_pid"][i]), G.e_kin(s["had_pid"][i], float(s["had_p"][i, 0])))
                 for i in range(h0, h1)]
        parts = [(pid, E) for pid, E in parts if E > 0]
        vis = sum(E for _, E in parts)
        miss = sum(E for pid, E in parts if pid not in lib)
        for pid, E in parts:
            if pid not in lib:
                missing_pids[pid] = missing_pids.get(pid, 0.0) + E
        if vis <= 0 or miss / vis > max_missing:
            n_skip += 1
            continue
        parts = [(pid, E) for pid, E in parts if pid in lib]
        if ch == "cc":                                   # G4 event = hadrons + electron
            parts.append((11, G.e_kin(11, float(s["lep"][e, 0]))))
        g4 = [g4_composite(EV, lib, parts, len(x), rng) for _ in range(n_g4 + 1)]
        ref, extra = g4[:n_g4], g4[n_g4]                 # 100 references + 1 for the floor

        E_had = float(s["blob"][e, 0])
        # the blob contains the struck nucleon's rest mass (baryon number 1), which
        # makes no light -> develop the analytic with the blob's kinetic energy
        E_had_kin = max(E_had - M_NUCLEON, 1e-3)
        ana = analytic_had(x, E_had_kin, mean_yield(model, "pip", E_had_kin))
        out = output_profile(s, e, x)                    # full event, incl. the electron
        # Units: the model's amplitudes were FITTED to the G4 per-bin profiles
        # (shower_gamma_model.fit_profile: curve_fit(A*kernel(x), y)), so
        # amp * unit-area kernel evaluated at the bin centres is already in G4 units.
        # The analytic uses the same convention (model yield * unit-area kernel).
        r = [E_had, float(s["E_nu"][e])]
        refb = blur(np.array(ref), sigma, binw)          # blur the 100 references ONCE
        for test in (out, ana, extra):
            l2, ks = l2_ks_many(refb, test)
            r += [float(np.nanmean(l2)), float(np.nanmean(ks))]
        rows.append(r)
        print(f"  {ch}: event {j + 1}/{len(events)}", end="\r")
    print(f"\n  {ch}: scored {len(rows)} events, skipped {n_skip} "
          f"(> {max_missing:.0%} of the hadronic energy in species without a G4 library)")
    if missing_pids:
        tot = sum(missing_pids.values())
        print("  energy in species with no G4 library: " +
              ", ".join(f"{pid}: {E / tot:.0%}" for pid, E in
                        sorted(missing_pids.items(), key=lambda t: -t[1])))
    return np.array(rows, float)


def l2ks_plot(rows, title, xlabel, path, keys=CURVE_ORDER):
    """rows: E, then (l2, ks) for each curve in CURVE_ORDER."""
    if len(rows) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)
    for ax, off in ((ax1, 0), (ax2, 1)):
        for c, key in enumerate(CURVE_ORDER):
            if key not in keys:
                continue
            lab, col, mk, ls = CURVES[key]
            y = rows[:, 1 + 2 * c + off]
            ok = np.isfinite(y) & (y > 0)
            if mk is None:
                ax.plot(rows[ok, 0], y[ok], ls, color=col, lw=2.0, alpha=0.9, label=lab)
            else:
                ax.plot(rows[ok, 0], y[ok], ls, color=col, lw=2.6, marker=mk, ms=9,
                        markeredgecolor="white", markeredgewidth=1.3, alpha=0.85, label=lab)
        ax.set_yscale("log")
        G.style(ax)
        G.legend(ax)
    ax1.set_ylabel(r"$L_2$ ($\sum(\mathrm{G4}-\mathrm{model})^2/\sum\mathrm{G4}^2$)", fontsize=13)
    ax2.set_ylabel("KS statistic", fontsize=14)
    ax2.set_xscale("log")
    ax2.set_xlabel(xlabel, fontsize=15)
    G.title(ax1, title)
    G.savefig(fig, os.path.dirname(path), os.path.basename(path))
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--outdir", default="output/gamma_dis_plots/ccnc")
    ap.add_argument("--energies", type=float, nargs="+", default=[1e3, 1e4, 1e5],
                    help="E_nu of the fixed-E_nu samples to plot")
    ap.add_argument("--ehad", type=float, nargs="+", default=[1e3, 1e4],
                    help="target E_had [GeV] for the fixed-E_had spaghetti plots")
    ap.add_argument("--tol", type=float, default=0.10, help="E_had window: +-tol (fractional)")
    ap.add_argument("--n", type=int, default=100, help="events per channel in F1/F2")
    ap.add_argument("--xlim", type=float, nargs=2, default=[0, 1600])
    ap.add_argument("--g4-dir", nargs="+", default=None,
                    help="dir(s) of shower_*_E*GeV.h5, merged (enables F3)")
    ap.add_argument("--n-g4", type=int, default=100,
                    help="G4 composites per event that each shower is compared with")
    ap.add_argument("--model", default=os.path.join(
        G.SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.npz"))
    ap.add_argument("--max-events", type=int, default=2000, help="events per channel in F3")
    ap.add_argument("--max-missing", type=float, default=0.01,
                    help="skip events with more than this energy fraction lacking G4 showers")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*outside the trained range.*")
    rng = np.random.default_rng(args.seed)
    load = lambda name: G.load_sample(os.path.join(args.samples, name))

    # F1: fixed E_nu
    for E in args.energies:
        groups = []
        for ch in ("cc", "nc"):
            s = load(f"ccnc_mono_E{E:.0e}_{ch}.npz")
            idx = pick(rng, np.arange(len(s["E_nu"])), args.n)
            groups.append((ch, s, idx, f"{G.elabel(E)} {ch.upper()} ({len(idx)} events)"))
        spaghetti(groups, rf"Event Showers at $E_\nu$ = {G.elabel(E)}",
                  os.path.join(args.outdir, f"ccnc_Enu_{E:.0e}.png"), args.xlim)

    # F2: fixed E_had (selected from the range samples)
    rng_s = {ch: load(f"ccnc_range_{ch}.npz") for ch in ("cc", "nc")}
    for Et in args.ehad:
        groups = []
        for ch, s in rng_s.items():
            Eh = s["blob"][:, 0]
            win = np.where(np.abs(np.log(Eh / Et)) < np.log1p(args.tol))[0]
            idx = pick(rng, win, args.n)
            print(f"E_had ~ {G.elabel(Et)}: {ch.upper()} has {len(win)} events in the window, "
                  f"plotting {len(idx)}")
            if len(idx):
                groups.append((ch, s, idx, rf"$E_\mathrm{{had}}\approx$ {G.elabel(Et)} "
                                           f"{ch.upper()} ({len(idx)} events)"))
        if groups:
            spaghetti(groups, rf"Event Showers at $E_\mathrm{{had}}\approx$ {G.elabel(Et)}",
                      os.path.join(args.outdir, f"ccnc_Ehad_{Et:.0e}.png"), args.xlim)

    # F3: L2 / KS vs G4
    if args.g4_dir is None:
        print("no --g4-dir: skipping the L2/KS vs G4 figures")
        return
    EV, C, lib = load_g4_tools([os.path.expanduser(d) for d in args.g4_dir])
    have = sorted(lib)
    print(f"G4 library species: {have}")
    need = [111, 211, -211, 321, -321, 310, 130, 2212, 2112]
    if [p for p in need if p not in lib]:
        print(f"  WARNING: no G4 library for {[p for p in need if p not in lib]}")
    if 11 not in lib:
        raise SystemExit("no electron (shower_em_*) G4 library -- needed for the CC events")
    any_pid = next(iter(lib)); any_E = next(iter(lib[any_pid]))
    x = np.asarray(lib[any_pid][any_E]["z_centers"], float)
    binw = float(x[1] - x[0])
    sigma = 0.0 if args.no_blur else C.DEPTH_RES_CM
    model = hs.load_model(args.model)
    for ch, s in rng_s.items():
        rows = score_channel(EV, lib, model, s, ch, x, sigma, binw, rng,
                             args.max_events, args.max_missing, args.n_g4)
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, f"ccnc_l2ks_{ch}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["E_had_GeV", "E_nu_GeV"] +
                       [f"{m}_{k}" for k in CURVE_ORDER for m in ("l2", "ks")])
            w.writerows(rows.tolist())
        if len(rows) == 0:
            continue
        vals = rows[:, 2:].T
        for col, xl, tag in ((0, r"Hadronic Energy  $E_\mathrm{had}$  [GeV]", "Ehad"),
                             (1, r"Neutrino Energy  $E_\nu$  [GeV]", "Enu")):
            l2ks_plot(binned(rows[:, col], vals),
                      f"{ch.upper()} Event Showers vs Geant4",
                      xl, os.path.join(args.outdir, f"ccnc_l2ks_{ch}_vs_{tag}.png"))


if __name__ == "__main__":
    main()
