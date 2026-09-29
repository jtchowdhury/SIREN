"""
compare_event_vs_g4.py
======================
Event-level version of compare_sampled_vs_g4.py: instead of scoring a single
hadron's shower, score the COMPOSITE shower of a whole neutrino-DIS event, as a
function of the event's total hadronic energy E_had.

For each PYTHIA event (from pythia_dis_secondaries.h5):
  * take the TOP-K most energetic model-known hadrons (K=10 by default),
  * lump ALL remaining hadronic energy (E_had - sum of those K) into ONE pi0,
  * build the composite longitudinal profile three/four ways:
      - Analytic (fixed shape) : SIREN placeholder gamma per hadron, scaled by
                                 the model's mean yield at that energy
      - Sampled Single Gamma   : m=1 draw per hadron            (optional, --with-single)
      - Gamma Mixture Model    : full sampled draw per hadron
      - G4 (truth)             : a real Geant4 shower per hadron
  * score each sampled composite against a G4 composite with the same metric as
    the single-hadron plot: relative L2 = sum((g4-model)^2)/sum(g4^2) and the KS
    gap of the normalized cumulative profiles, with the G4 side blurred by the
    detector depth resolution (~45 cm).
  * also score a G4-vs-G4 pair (two independent G4 composites of the SAME event)
    to get the empirical floor.

Because Geant4 only simulated discrete energies {10,30,...,30000 GeV} but the
hadrons here have arbitrary energies, the G4 shower for a hadron of energy E is
drawn from the NEAREST simulated energy and rescaled so its total light matches
E (yield ~ E).  This is the one approximation on the G4 side; if it proves too
coarse we can later simulate G4 showers at the actual E_had values and swap out
`_g4_draw` -- everything else stays the same.

Events are pooled across all E_nu groups and binned in log-E_had; the mean L2
and KS per bin are plotted vs E_had (2 panels), mirroring the single-hadron
figure.

Run on the cluster, e.g.:
    python compare_event_vs_g4.py \
        --g4-dir  ../output \
        --model   ../output/shower_model.pkl \
        --pythia  ../../resources/analysis/output/pythia_dis_secondaries.h5 \
        --outdir  ../output/plots/result
"""
import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter1d

from shower_gamma_model import (
    load_model, ShowerSampler, load_g4_library, _kernel,
    NAME_TO_PID, PID_TO_NAME,
)
import compare_methods as C

SPECIES_LATEX = C.SPECIES_LATEX

# curve styles (match the single-hadron figure)
META = {
    "analytic": ("Analytic (fixed shape)", "#4682B4"),
    "single":   ("Sampled Single Gamma",   "#9B4DCA"),
    "mixture":  ("Gamma Mixture Model",     "#F08070"),
    "floor":    ("G4 vs G4 (empirical floor)", "#444444"),
}


# ---------------------------------------------------------------------------
#  model backend: sample from a .pkl (shower_gamma_model) OR a .npz
#  (hadronic_showers runtime). Either way the G4 side reads the h5 library.
# ---------------------------------------------------------------------------
class _PklModel:
    supports_single = True

    def __init__(self, path):
        self.interp = load_model(path)
        self.sampler = ShowerSampler(self.interp)

    def sample_mixture(self, pid, E, x, rng):
        return self.sampler.sample_profile(pid, E, rng, x=x)[0]

    def sample_single(self, pid, E, x, rng):
        r = C.method2_sample(self.interp, pid, E, x, rng)
        return None if r is None else r[0]

    def yield_mean(self, pid, E):
        return self.interp.yield_mean(pid, E)


class _NpzModel:
    supports_single = False        # the m=1-only pool isn't stored in the npz

    def __init__(self, path):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "python"))
        import hadronic_showers as hs
        self.m = hs.load_model(path)

    def sample_mixture(self, pid, E, x, rng):
        return self.m.sample(pid, E, rng, x=x).photons

    def sample_single(self, pid, E, x, rng):
        return None

    def yield_mean(self, pid, E):
        return self.m.yield_mean(pid, E)


def _load_backend(path):
    return _NpzModel(path) if str(path).endswith(".npz") else _PklModel(path)


# ---------------------------------------------------------------------------
#  per-hadron shower builders
# ---------------------------------------------------------------------------
def _analytic_profile(x, E, yield_tot):
    """SIREN placeholder: a gamma in radiation lengths, scaled to a total light
    `yield_tot` (we use the model's mean yield at E, since there is no G4 mean at
    an arbitrary energy to normalise against)."""
    alpha = 0.3 + 0.7 * np.log(E / C.HAD_EC_GEV)
    beta = 0.9
    t = np.asarray(x, float) / C.X0_ICE_CM
    shape = _kernel(t, alpha, beta)
    s = shape.sum()
    return shape / s * yield_tot if s > 0 else shape


def _g4_draw(lib, pid, E, rng):
    """A Geant4 shower for a hadron of (arbitrary) energy E: take a random shower
    at the NEAREST simulated energy and rescale its total light to E (yield ~ E).
    Returns None if this species has no G4 library entry."""
    if pid not in lib or not lib[pid]:
        return None
    grid = np.array(sorted(lib[pid].keys()), float)
    Eg = float(grid[np.argmin(np.abs(np.log(grid) - np.log(E)))])   # nearest in log-E
    profs = lib[pid][Eg]["profiles"]
    p = np.asarray(profs[rng.integers(len(profs))], float)
    return p * (E / Eg)


# ---------------------------------------------------------------------------
#  event parsing:  top-K hadrons + one remainder pi0
# ---------------------------------------------------------------------------
def _event_final_state(pids, energies, E_had, rng, top_k=10):
    """[(species, E_GeV), ...]: the top_k most energetic model-known hadrons plus
    one pi0 carrying all the leftover hadronic energy (E_had - sum of those k)."""
    hadrons = []
    for pid, e in zip(pids, energies):
        pid = int(pid)
        if pid == 0 or not np.isfinite(e) or e <= 0:
            continue
        name = PID_TO_NAME.get(pid)
        if name is None and abs(pid) == 311:            # K0/K0bar -> KS/KL
            name = "KS" if rng.random() < 0.5 else "KL"
        if name in (None, "em", "ep"):                  # skip non-hadrons / leptons
            continue
        hadrons.append((name, float(e)))
    hadrons.sort(key=lambda t: -t[1])
    top = hadrons[:top_k]
    rem = max(float(E_had) - sum(e for _, e in top), 0.0)
    fs = list(top)
    if rem > 0:
        fs.append(("pi0", rem))
    return fs


# ---------------------------------------------------------------------------
#  composites on a common depth grid
# ---------------------------------------------------------------------------
def _composite(fs, x, how, model, lib, rng):
    """Sum a per-hadron profile over the final state. `how` selects the builder."""
    total = np.zeros_like(x)
    for name, E in fs:
        pid = NAME_TO_PID.get(name)
        if pid is None:
            continue
        if how == "mixture":
            p = model.sample_mixture(pid, E, x, rng)
        elif how == "single":
            p = model.sample_single(pid, E, x, rng)
            if p is None:
                continue
        elif how == "analytic":
            p = _analytic_profile(x, E, model.yield_mean(pid, E))
        elif how == "g4":
            p = _g4_draw(lib, pid, E, rng)
            if p is None:
                continue
        else:
            raise ValueError(how)
        total = total + p
    return total


def _l2_ks(g4, model, sigma_cm, binw):
    """Relative L2 and KS gap of one (G4, model) composite pair; the G4 side is
    blurred by the detector depth resolution. Identical math to the single-hadron
    sampled_vs_g4()."""
    g = gaussian_filter1d(g4, max(sigma_cm / binw, 1e-6), mode="constant") \
        if sigma_cm > 0 else np.asarray(g4, float)
    s = np.asarray(model, float)
    denom = float(np.dot(g, g))
    if denom <= 0:
        return np.nan, np.nan
    l2 = float(np.sum((g - s) ** 2) / denom)
    gp, sp = np.clip(g, 0, None), np.clip(s, 0, None)
    gs, ss = gp.sum(), sp.sum()
    ks = float(np.max(np.abs(np.cumsum(gp) / gs - np.cumsum(sp) / ss))) \
        if gs > 0 and ss > 0 else np.nan
    return l2, ks


# ---------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------
def load_events(pythia_h5):
    """All events across all E_nu groups -> arrays (E_had, pids[20], energies[20])."""
    import h5py
    E_had, PIDS, ENER = [], [], []
    with h5py.File(pythia_h5, "r") as f:
        for key in sorted(f):
            g = f[key]
            eh = np.asarray(g["E_had"])
            pd = np.asarray(g["top20_pids"])
            en = np.asarray(g["top20_energies"])
            E_had.append(eh); PIDS.append(pd); ENER.append(en)
    return (np.concatenate(E_had), np.concatenate(PIDS, axis=0),
            np.concatenate(ENER, axis=0))


def run(args):
    import warnings
    # soft secondaries fall below each species' trained energy floor; the model
    # clamps them to the nearest trained energy. That's expected here, so silence
    # the per-hadron warning (it would otherwise fire thousands of times).
    warnings.filterwarnings("ignore", message=r".*outside the trained range.*")
    print("note: hadron energies outside a species' trained range are clamped to "
          "the nearest trained energy (expected for soft secondaries).")
    model = _load_backend(args.model)
    if args.with_single and not model.supports_single:
        print("  note: --with-single needs a .pkl model (the m=1-only pool isn't in "
              "the .npz); dropping the Single Gamma curve.")
        args.with_single = False
    lib = load_g4_library(args.g4_dir)
    have = sorted(PID_TO_NAME.get(p, str(p)) for p in lib)
    print(f"G4 library species ({len(have)}): {have}")
    missing = [n for n in ("pi0", "pip", "pim", "Kp", "Km", "KS", "KL", "p", "n")
               if n not in have]
    if missing:
        print(f"  WARNING: G4 library is missing {missing}; events whose hadrons "
              f"are all missing get no G4 composite and are skipped.")
    # common depth grid (all species share the detector binning)
    any_pid = next(iter(lib)); any_E = next(iter(lib[any_pid]))
    x = np.asarray(lib[any_pid][any_E]["z_centers"], float)
    binw = float(x[1] - x[0])
    sigma = 0.0 if args.no_blur else args.depth_res_cm

    E_had, PIDS, ENER = load_events(args.pythia)
    print(f"loaded {len(E_had)} events; E_had {E_had.min():.1f}-{E_had.max():.1f} GeV")

    # log-E_had bins, pooled over all E_nu groups
    lo, hi = max(E_had.min(), 1.0), E_had.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), args.n_bins + 1)
    which = np.digitize(E_had, edges) - 1

    forms = ["analytic", "mixture", "floor"] + (["single"] if args.with_single else [])
    rng = np.random.default_rng(args.seed)
    rows = []            # (E_center, form, l2, ks)
    n_nog4 = 0           # events dropped for lack of any G4 truth

    for b in range(args.n_bins):
        idx = np.where(which == b)[0]
        if len(idx) < args.min_count:
            continue
        if len(idx) > args.n_per_bin:
            idx = rng.choice(idx, args.n_per_bin, replace=False)
        acc = {f: {"l2": [], "ks": []} for f in forms}
        used = []
        for i in idx:
            fs = _event_final_state(PIDS[i], ENER[i], E_had[i], rng, args.top_k)
            if not fs:
                continue
            g4A = _composite(fs, x, "g4", model, lib, rng)
            if not np.any(g4A > 0):        # no G4 truth for this event -> skip
                n_nog4 += 1
                continue
            used.append(i)
            for f in forms:
                comp = (_composite(fs, x, "g4", model, lib, rng)
                        if f == "floor" else
                        _composite(fs, x, f, model, lib, rng))
                l2, ks = _l2_ks(g4A, comp, sigma, binw)
                acc[f]["l2"].append(l2); acc[f]["ks"].append(ks)
        if not used:
            print(f"  bin {b:2d}: skipped (no scorable events)")
            continue
        Ecen = float(np.exp(np.mean(np.log(E_had[used]))))     # geo-mean E_had in bin
        for f in forms:
            rows.append((Ecen, f, _nanmean(acc[f]["l2"]), _nanmean(acc[f]["ks"])))
        print(f"  bin {b:2d}: E_had~{Ecen:8.1f} GeV  n={len(used)}")

    if n_nog4:
        print(f"  ({n_nog4} events skipped for lack of any G4-library shower)")
    if not rows:
        raise SystemExit("no scorable bins -- check that the G4 library in --g4-dir "
                         "actually contains the hadron species.")
    _plot(rows, forms, args, sigma)
    _write_csv(rows, os.path.join(args.outdir, "event_vs_g4.csv"))


def _write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["E_had_GeV", "form", "l2", "ks"])
        for r in rows:
            w.writerow(r)
    print("wrote", path)


def _nanmean(a):
    """Mean of the finite values; NaN if none (no RuntimeWarning on empty/all-NaN)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _series(rows, form, k):
    pts = sorted((E, l2, ks) for (E, f, l2, ks) in rows if f == form)
    E = np.array([p[0] for p in pts])
    v = np.array([p[1 if k == "l2" else 2] for p in pts])
    ok = np.isfinite(E) & np.isfinite(v) & (v > 0)          # log-axis safe
    return E[ok], v[ok]


def _plot(rows, forms, args, sigma):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.4), sharex=True)

    order = [f for f in ("analytic", "single", "mixture") if f in forms] + ["floor"]

    def draw(ax, k):
        any_pts = False
        for f in order:
            label, color = META[f]
            E, v = _series(rows, f, k)
            if len(E) == 0:
                continue
            any_pts = True
            if f == "floor":
                ax.plot(E, v, "--", color=color, lw=2.0, alpha=0.9, label=label)
            else:
                ax.plot(E, v, "-", color=color, lw=2.6, marker="o", ms=9,
                        markeredgecolor="white", markeredgewidth=1.3,
                        alpha=0.85, label=label)
        ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=12, length=6)
        return any_pts

    has1 = draw(ax1, "l2")
    if has1:
        ax1.set_yscale("log")
    ax1.set_ylabel(r"$L_2$ ($\sum(\mathrm{data}-\mathrm{model})^2/\sum\mathrm{data}^2$)",
                   fontsize=13)
    note = "  [unblurred G4]" if sigma <= 0 else ""
    ax1.set_title(f"Sampled vs Simulated (G4) Event Shower",
                  fontsize=15, fontweight="bold", pad=10)
    ax1.legend(fontsize=11, framealpha=0.92, loc="best")

    has2 = draw(ax2, "ks")
    ax2.set_xscale("log")
    if has2:
        ax2.set_yscale("log")
    ax2.set_ylabel("KS statistic", fontsize=14)
    ax2.set_xlabel(r"Hadronic Shower Energy  $E_\mathrm{had}$  [GeV]", fontsize=15)
    ax2.legend(fontsize=11, framealpha=0.92, loc="best")

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "event_vs_g4.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g4-dir", default="../output", help="dir of shower_*_E*GeV.h5")
    ap.add_argument("--model", default="../output/shower_model.pkl",
                    help="model file: a .pkl (shower_gamma_model) OR a .npz "
                         "(hadronic_showers runtime). .npz can't draw the single-gamma curve.")
    ap.add_argument("--pythia", required=True, help="pythia_dis_secondaries.h5")
    ap.add_argument("--outdir", default="../output/plots/result")
    ap.add_argument("--top-k", type=int, default=10, help="hadrons kept per event")
    ap.add_argument("--n-bins", type=int, default=12, help="log-E_had bins")
    ap.add_argument("--n-per-bin", type=int, default=200,
                    help="max events scored per bin (subsampled if more)")
    ap.add_argument("--min-count", type=int, default=20,
                    help="skip bins with fewer events than this")
    ap.add_argument("--with-single", action="store_true",
                    help="also plot the Sampled Single Gamma curve (off by default)")
    ap.add_argument("--no-blur", action="store_true",
                    help="compare against RAW (unblurred) G4")
    ap.add_argument("--depth-res-cm", type=float, default=C.DEPTH_RES_CM,
                    help="detector depth resolution for the G4 blur (~45 cm)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
