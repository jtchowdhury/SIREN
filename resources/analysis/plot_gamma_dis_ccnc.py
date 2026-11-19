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
    For every SIREN event, --n-g4 (100) SIMULATED event showers are built. Each one:
      * hadrons: with --reference library (default), a random full PYTHIA DIS event
        from the event library with (nearly) this event's E_had, energies scaled to
        E_had; with --reference same, SIREN's own hadron list for this event;
      * one stored Geant4 shower per hadron (at its kinetic energy), plus one for
        the electron in CC, summed.
    "library" tests hadronization + showers against PYTHIA DIS + Geant4; "same"
    tests only the showers, for identical hadrons. Compared with each of the 100:
      * Sampled Gamma Shower: the event's output subshowers (all, incl. the electron);
      * Fixed Analytical Shower: one analytic shower of the hadronic blob only;
      * floor: one extra simulated shower.
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
CURVES = {"gamma":    ("Sampled Gamma Shower vs simulation", "#F08070", "o", "-"),
          "analytic": ("Fixed Analytical Shower vs simulation", "#4682B4", "s", "-"),
          "floor":    ("simulation vs simulation (floor)", "#444444", None, "--")}
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


class PythiaLibrary:
    """Full PYTHIA DIS events (top-20 hadrons each), sorted in E_had, to draw the
    hadron list of a simulated event at a given E_had."""

    def __init__(self, path, max_events=300000, seed=0):
        with np.load(path) as d:
            if "top20_ekin" not in d.files:
                raise SystemExit(f"{path} was written before kinetic energies were stored: "
                                 "regenerate it with pythia_dis_reference.py library")
            Eh, pids, ek, rest = d["E_had"], d["top20_pids"], d["top20_ekin"], d["rest_ekin"]
        if len(Eh) > max_events:
            keep = np.random.default_rng(seed).choice(len(Eh), max_events, replace=False)
            Eh, pids, ek, rest = Eh[keep], pids[keep], ek[keep], rest[keep]
        o = np.argsort(Eh)
        self.logE, self.Eh, self.pids = np.log(Eh[o]), Eh[o], pids[o]
        self.ek, self.rest = ek[o], rest[o]

    def draw(self, E_had, rng, g4lib, window=50):
        """[(pid, kinetic energy)] of a library event near E_had, scaled to E_had.
        K0 -> KS/KL; the kinetic energy of the hadrons beyond the stored 20 and of
        species without a G4 shower goes into one pi0, as in GammaDISFromSpline."""
        i = int(np.searchsorted(self.logE, np.log(E_had)))
        lo = max(0, min(i - window // 2, len(self.Eh) - window))
        k = int(rng.integers(lo, min(lo + window, len(self.Eh))))
        scale = E_had / self.Eh[k]
        parts, rest = [], float(self.rest[k]) * scale
        for pid, Ek in zip(self.pids[k], self.ek[k] * scale):
            pid, Ek = int(pid), float(Ek)
            if pid == 0 or Ek <= 0:
                continue
            if abs(pid) == 311:
                pid = 310 if rng.random() < 0.5 else 130
            if pid in g4lib:
                parts.append((pid, Ek))
            else:
                rest += Ek
        if rest > 1e-9 * E_had and 111 in g4lib:
            parts.append((111, rest))
        return parts


def score_channel(EV, lib, model, s, ch, x, sigma, binw, rng, max_events, max_missing, n_g4,
                  pylib=None):
    n_ev = len(s["E_nu"])
    events = pick(rng, np.arange(n_ev), max_events)
    rows, n_skip, missing_pids = [], 0, {}
    for j, e in enumerate(events):
        h0, h1 = s["had_off"][e], s["had_off"][e + 1]
        # hadrons at kinetic energy (the G4 library / model are labelled by E_kin)
        parts = [(int(s["had_pid"][i]), G.hadron_ekin(s, i)) for i in range(h0, h1)]
        parts = [(pid, E) for pid, E in parts if E > 0]
        vis = sum(E for _, E in parts)
        miss = sum(E for pid, E in parts if pid not in lib)
        for pid, E in parts:
            if pid not in lib:
                missing_pids[pid] = missing_pids.get(pid, 0.0) + E
        if vis <= 0 or (pylib is None and miss / vis > max_missing):
            n_skip += 1
            continue
        E_had = float(s["blob"][e, 0])
        lepton = [(11, G.e_kin(11, float(s["lep"][e, 0])))] if ch == "cc" else []
        if pylib is None:                                # SIREN's own hadrons for this event
            parts = [(pid, E) for pid, E in parts if pid in lib] + lepton
            g4 = [g4_composite(EV, lib, parts, len(x), rng) for _ in range(n_g4 + 1)]
        else:                                            # a new PYTHIA DIS event each time
            g4 = [g4_composite(EV, lib, pylib.draw(E_had, rng, lib) + lepton, len(x), rng)
                  for _ in range(n_g4 + 1)]
        ref, extra = g4[:n_g4], g4[n_g4]                 # 100 references + 1 for the floor

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
    print(f"\n  {ch}: scored {len(rows)} events, skipped {n_skip}")
    if missing_pids:
        tot = sum(missing_pids.values())
        print("  energy in species with no G4 library: " +
              ", ".join(f"{pid}: {E / tot:.0%}" for pid, E in
                        sorted(missing_pids.items(), key=lambda t: -t[1])))
    return np.array(rows, float)


def concat_samples(samples):
    """Join samples written by gamma_dis_ccnc_samples.py (offset arrays are re-based)."""
    out = {}
    for k in samples[0]:
        if np.ndim(samples[0][k]) == 0:
            continue
        if k in ("had_off", "sub_off"):
            parts, shift = [], 0
            for s in samples:
                parts.append(s[k][:-1] + shift)
                shift += s[k][-1]
            out[k] = np.concatenate(parts + [np.array([shift])])
        else:
            out[k] = np.concatenate([s[k] for s in samples])
    return out


def by_energy(E, vals):
    """Means of each column for every distinct energy (fixed-energy samples)."""
    keys = np.round(np.log10(E), 3)
    rows = []
    for k in np.unique(keys):
        sel = keys == k
        rows.append([float(np.median(E[sel]))] + [float(np.nanmean(v[sel])) for v in vals])
    return np.array(rows)


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
    ap.add_argument("--source", choices=["mono", "range"], default="mono",
                    help="events for the L2/KS plot: the fixed-E_nu samples (default; plotted "
                         "vs E_nu only) or the spread-energy samples (also vs E_had, slow to make)")
    ap.add_argument("--l2ks-energies", type=float, nargs="+", default=list(np.logspace(2, 6, 9)),
                    help="E_nu of the fixed-E_nu samples used with --source mono")
    ap.add_argument("--reference", choices=["library", "same"], default="library",
                    help="hadrons of the simulated showers: PYTHIA DIS library events at the "
                         "same E_had (default), or SIREN's own hadron list")
    ap.add_argument("--pythia-lib-cc", default="output/pythia_events_cc.npz",
                    help="CC event library from pythia_dis_reference.py library --channel cc")
    ap.add_argument("--pythia-lib-nc", default="output/pythia_events_nc.npz",
                    help="NC event library from pythia_dis_reference.py library --channel nc")
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
    if args.source == "range":
        rng_s = {ch: load(f"ccnc_range_{ch}.npz") for ch in ("cc", "nc")}
    else:
        rng_s = {ch: concat_samples([load(f"ccnc_mono_E{E:.0e}_{ch}.npz") for E in args.l2ks_energies])
                 for ch in ("cc", "nc")}
    for Et in (args.ehad if args.source == "range" else []):
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
    pylibs = {"cc": None, "nc": None}
    if args.reference == "library":
        for ch in ("cc", "nc"):
            path = os.path.expanduser(getattr(args, f"pythia_lib_{ch}"))
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found: run pythia_dis_reference.py library --channel {ch} "
                                 f"--out {path}")
            pylibs[ch] = PythiaLibrary(path)
            print(f"{ch.upper()}: simulated showers use {len(pylibs[ch].Eh)} PYTHIA DIS events from {path}")
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
                             args.max_events, args.max_missing, args.n_g4, pylibs[ch])
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, f"ccnc_l2ks_{ch}_ref-{args.reference}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["E_had_GeV", "E_nu_GeV"] +
                       [f"{m}_{k}" for k in CURVE_ORDER for m in ("l2", "ks")])
            w.writerows(rows.tolist())
        if len(rows) == 0:
            continue
        vals = rows[:, 2:].T
        axes = [(1, r"Neutrino Energy  $E_\nu$  [GeV]", "Enu")]
        if args.source == "range":
            axes.insert(0, (0, r"Hadronic Energy  $E_\mathrm{had}$  [GeV]", "Ehad"))
        for col, xl, tag in axes:
            pts = by_energy(rows[:, 1], vals) if args.source == "mono" else binned(rows[:, col], vals)
            l2ks_plot(pts,
                      f"{ch.upper()} Event Showers vs {'PYTHIA + ' if pylibs[ch] else ''}Geant4",
                      xl, os.path.join(args.outdir, f"ccnc_l2ks_{ch}_ref-{args.reference}_vs_{tag}.png"))


if __name__ == "__main__":
    main()
