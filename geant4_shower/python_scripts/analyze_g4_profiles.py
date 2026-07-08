"""
analyze_g4_profiles.py
----------------------
Visualization and analysis of the Geant4 Cherenkov profile library.

Produces three figures:

  1. Profile library   — mean ± 1σ longitudinal Cherenkov profile for each
                         species at all simulated energies (one panel per species).
  2. Cumulative k plot — how many sub-cascades are needed to capture 90% of
                         Cherenkov yield.Requires the Pythia DIS file 
                         (for sub-cascade species and energies); 
                         uses G4 N_total via log-linear interpolation.                         
  3. Yield curves      — mean total Cherenkov photon count (N_total) vs energy
                         per species (log-log). The G4 counterpart to the Pythia
                         multiplicity plot — shows which species contributes most.
  4. Individual runs   — 10 individual G4 runs of a single (species, energy),
                         overlaid on one axis, to show run-to-run fluctuation
                         and multimodality.
  5. Composite shower  — for each Pythia DIS event, sum the mean G4 Cherenkov

Run from geant4_shower/python_scripts/ (or anywhere — paths are absolute):
    python analyze_g4_profiles.py

Paths:
    G4 output  : geant4_shower/output/shower_<name>_E<E>GeV.h5
    Pythia data: resources/analysis/output/pythia_dis_secondaries.h5
    Plots out  : geant4_shower/output/g4_*.png
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# ── Paths (resolved relative to this script's location) ───────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
G4_ROOT     = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))          # geant4_shower/
G4_DIR      = os.path.join(G4_ROOT, "output")                          # geant4_shower/output/
OUT_DIR     = os.path.join(G4_ROOT, "output")                          # plots go here too
PYTHIA_FILE = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../resources/analysis/output/pythia_dis_secondaries.h5")
)
os.makedirs(OUT_DIR, exist_ok=True)

# E_nu group in Pythia data to skip (no reliable G4 coverage above 30 TeV)
SKIP_PYTHIA_GROUPS = {"E_nu_1e+05"}

# ── Species metadata ──────────────────────────────────────────────────────────
SPECIES = [
    (111,   "pi0",  "π⁰",   "steelblue"),
    (211,   "pip",  "π⁺",   "darkorange"),
    (-211,  "pim",  "π⁻",   "tomato"),
    (321,   "Kp",   "K⁺",   "green"),
    (-321,  "Km",   "K⁻",   "purple"),
    (310,   "KS",   "K_S",  "olive"),
    (130,   "KL",   "K_L",  "teal"),
    (2212,  "p",    "p",    "sienna"),
    (2112,  "n",    "n",    "navy"),
]
PID_TO_META = {pid: (name, label, color) for pid, name, label, color in SPECIES}

# PDG IDs present in Pythia top20 but not in our G4 library → proxy mapping
# KS(310), KL(130), p(2212), n(2112) all have real G4 data up to 30 TeV.
# Only antiparticles lack data (rare in ν CC DIS final states).
PID_PROXY = {
    -2212: 2212,  # p̄  → p   (same shower physics)
    -2112: 2112,  # n̄  → n
}

# ── Plot flags — set False to skip ────────────────────────────────────────────
PLOT_PROFILE_LIBRARY  = True   # Plot 1: longitudinal profile per species
PLOT_CUMULATIVE_K     = True   # Plot 2: sub-cascades needed for 90% Cherenkov yield
PLOT_YIELD_CURVES     = True   # Plot 3: N_total vs energy per species
PLOT_COMPOSITE_SHOWER = True   # Plot 4: composite hadronic shower profile (Pythia DIS)
PLOT_INDIVIDUAL_RUNS   = True  # Plot 5: 10 individual 1 TeV pi+ runs (illustrate multimodality)
PLOT_SAMPLED_COMPOSITE = True  # Plot 6: 10 sampled composite showers (nearest-E run + rescale)
PLOT_EVENT_SUBSHOWERS  = True  # Plot 7: per-event sub-shower breakdown + composite

# DIS proxy cut applied to Pythia events: E_had > this is necessary for W > 2 GeV.
# Replace with a real (Q2 > 1) & (W > 2) cut once the Pythia file stores Q2/W.
E_HAD_MIN_GEV = 2.6


def nice_label(grp_name):
    """'E_nu_1e+03' → '1 TeV' etc."""
    e = float(grp_name.replace("E_nu_", ""))
    if e >= 1e6: return f"{e/1e6:.0f} PeV"
    if e >= 1e3: return f"{e/1e3:.0f} TeV"
    return f"{e:.0f} GeV"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_g4_library():
    """
    Scan G4_DIR for shower_<name>_E<E>GeV.h5 files.
    Returns dict: {pid: {E_GeV: {"profiles": ndarray, "N_total": ndarray,
                                  "z_edges": ndarray}}}
    """
    pattern = os.path.join(G4_DIR, "shower_*_E*GeV.h5")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No G4 files found matching {pattern}")

    name_to_pid = {name: pid for pid, name, _, _ in SPECIES}

    library = {}
    for fpath in files:
        base = os.path.basename(fpath)
        m    = re.match(r"shower_(\w+)_E(\d+)GeV\.h5", base)
        if not m:
            continue
        name, e_str = m.group(1), m.group(2)
        pid = name_to_pid.get(name)
        if pid is None:
            continue
        E = float(e_str)

        with h5py.File(fpath, "r") as hf:
            profiles = hf["profiles"][:]
            N_total  = hf["N_total"][:]
            z_edges  = hf["z_edges"][:]

        library.setdefault(pid, {})[E] = {
            "profiles": profiles,
            "N_total":  N_total,
            "z_edges":  z_edges,
        }

    n_points = sum(len(v) for v in library.values())
    print(f"Loaded {n_points} (species, energy) points from {G4_DIR}")
    for pid, energies in library.items():
        name = PID_TO_META[pid][0]
        print(f"  {name:4s} ({pid:5d}): {sorted(energies.keys())} GeV")
    return library


def build_interpolators(library):
    """
    For each pid, build a log-log interpolator: E_GeV → mean N_total.
    Extrapolates in both directions using the log-log slope at the boundaries.
    Returns dict: {pid: callable(E_GeV) -> mean_N_total}
    """
    interpolators = {}
    for pid, e_data in library.items():
        energies = sorted(e_data.keys())
        means    = [e_data[E]["N_total"].mean() for E in energies]
        log_e    = np.log10(energies)
        log_n    = np.log10(means)
        interp   = interp1d(log_e, log_n, kind="linear",
                            fill_value="extrapolate", bounds_error=False)
        interpolators[pid] = interp
    return interpolators


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Profile library
# ─────────────────────────────────────────────────────────────────────────────

def plot_profile_library(library):
    """
    3×3 grid of panels (one per species). Each panel overlays mean ± 1σ
    longitudinal Cherenkov profiles at all simulated energies, color-coded
    by energy (log scale). Y-axis is log-scale, independent per species.
    """
    n_cols, n_rows = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharey=False)
    axes_flat = axes.flatten()
    cmap = cm.plasma

    for idx, (pid, name, label, base_color) in enumerate(SPECIES):
        ax = axes_flat[idx]

        if pid not in library:
            ax.set_title(f"{label}  (no data)")
            ax.set_visible(True)
            continue

        e_data    = library[pid]
        energies  = sorted(e_data.keys())
        log_e_min = np.log10(energies[0])
        log_e_max = np.log10(energies[-1])

        for E in energies:
            profiles  = e_data[E]["profiles"]
            z_edges   = e_data[E]["z_edges"]
            z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

            mean_p = profiles.mean(axis=0)
            std_p  = profiles.std(axis=0)

            t     = (np.log10(E) - log_e_min) / max(log_e_max - log_e_min, 1)
            color = cmap(0.15 + 0.75 * t)

            e_label = (f"{E/1000:.0f} TeV" if E >= 1000 else f"{E:.0f} GeV")
            ax.plot(z_centers, mean_p, color=color, linewidth=1.2, label=e_label)
            # σ band: clip to positive values for log scale
            lo = np.maximum(mean_p - std_p, 1e-1)
            ax.fill_between(z_centers, lo, mean_p + std_p, alpha=0.15, color=color)

        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.set_xlabel("Depth in ice [cm]", fontsize=8)
        ax.set_ylabel("Cherenkov photons / 5 cm", fontsize=8)
        ax.set_title(f"{label}", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        ax.set_xlim(0, None)

    # Hide any unused panels (none with 9 species and 3×3, but just in case)
    for idx in range(len(SPECIES), n_rows * n_cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Geant4 Cherenkov longitudinal profiles  (mean ± 1σ)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "g4_profile_library.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Cumulative k using real G4 yields
# ─────────────────────────────────────────────────────────────────────────────

def lookup_yield(pid, E_GeV, interpolators):
    """
    Return interpolated (or extrapolated) mean N_total for (pid, E_GeV).
    Returns 0 for padding entries (pid==0 or E<=0).
    Falls back to proxy pid if species not in library.
    Extrapolates in both directions using the log-log boundary slope.
    """
    if pid == 0 or E_GeV <= 0:
        return 0.0
    resolved = PID_PROXY.get(pid, pid)
    if resolved not in interpolators:
        return 0.0
    val = interpolators[resolved](np.log10(E_GeV))
    return float(10 ** val)


def plot_cumulative_k_g4(interpolators):
    """
    Recreates the cumulative Cherenkov fraction vs k plot using G4-simulated
    N_total values. E_nu groups in SKIP_PYTHIA_GROUPS are left as blank panels
    (no reliable G4 coverage for their sub-cascade energies).
    """
    if not os.path.exists(PYTHIA_FILE):
        print(f"Skipping cumulative-k plot: {PYTHIA_FILE} not found.")
        return

    with h5py.File(PYTHIA_FILE, "r") as pf:
        groups = sorted(pf.keys())

        fig, axes = plt.subplots(1, len(groups),
                                 figsize=(5 * len(groups), 4), sharey=True)
        if len(groups) == 1:
            axes = [axes]

        for ax, grp_name in zip(axes, groups):
            ax.set_title(f"E_ν = {nice_label(grp_name)}")
            ax.set_xlim(1, 20)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("# sub-cascades tracked (top k by energy)")

            grp        = pf[grp_name]
            top20_e    = grp["top20_energies"][:]   # (nevents, 20)
            top20_pids = grp["top20_pids"][:]       # (nevents, 20)
            E_had      = grp["E_had"][:]
            valid      = np.where(E_had > E_HAD_MIN_GEV)[0]
            top20_e    = top20_e[valid]
            top20_pids = top20_pids[valid]
            n_events, k_max = top20_e.shape

            # Look up G4 yield for each sub-cascade; drop events with any nan
            # (sub-cascade energy outside G4 range)
            g4_yields = np.array([
                [lookup_yield(int(top20_pids[i, k]), top20_e[i, k], interpolators)
                 for k in range(k_max)]
                for i in range(n_events)
            ], dtype=float)

            total     = g4_yields.sum(axis=1, keepdims=True)
            total     = np.where(total == 0, 1.0, total)
            cumfrac   = np.cumsum(g4_yields, axis=1) / total

            mean_cf = cumfrac.mean(axis=0)
            std_cf  = cumfrac.std(axis=0)
            k       = np.arange(1, k_max + 1)

            k90 = int(np.argmax(mean_cf >= 0.90)) + 1 if (mean_cf >= 0.90).any() else k_max

            ax.plot(k, mean_cf, "o-", color="steelblue", ms=4, label="mean")
            ax.fill_between(k, mean_cf - std_cf, mean_cf + std_cf,
                            alpha=0.25, color="steelblue", label="±1σ")
            ax.axhline(0.90, color="r", linestyle="--", linewidth=0.9,
                       label=f"90% (k={k90})")
            ax.axvline(k90, color="r", linestyle=":", linewidth=0.7, alpha=0.7)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Cumulative Cherenkov fraction")
        fig.suptitle(
            "How many sub-cascades to capture 90% of Cherenkov yield?\n"
            "(G4-simulated yields via interpolation, Pythia DIS sub-cascade energies)",
            fontsize=10)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, "g4_cumulative_cherenkov_fraction.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Cherenkov yield curves (N_total vs E per species)
# ─────────────────────────────────────────────────────────────────────────────

def plot_yield_curves(library):
    """
    Mean total Cherenkov photon yield (N_total) vs kinetic energy, per species.
    Left panel: absolute yield (log-log). Right panel: σ/μ (fluctuation strength).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for pid, name, label, color in SPECIES:
        if pid not in library:
            continue
        e_data   = library[pid]
        energies = sorted(e_data.keys())
        means    = np.array([e_data[E]["N_total"].mean() for E in energies])
        stds     = np.array([e_data[E]["N_total"].std()  for E in energies])
        rel_std  = stds / np.where(means > 0, means, 1.0)
        e_arr    = np.array(energies)

        ax1.errorbar(e_arr, means, yerr=stds, fmt="o-", color=color,
                     label=label, ms=4, capsize=3, linewidth=1.5)
        ax2.plot(e_arr, rel_std, "o-", color=color, label=label,
                 ms=4, linewidth=1.5)

    # Reference line: pure EM scaling N ∝ E
    if 111 in library:
        e_ref0 = sorted(library[111].keys())[0]
        n_ref0 = library[111][e_ref0]["N_total"].mean()
        e_ref  = np.array([10, 3e4])
        ax1.plot(e_ref, n_ref0 * e_ref / e_ref0, "k--", linewidth=0.8,
                 alpha=0.5, label="∝ E (EM reference)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Particle energy (kinetic ≈ total) [GeV]")
    ax1.set_ylabel("Mean N_total  (Cherenkov photons, 300–600 nm)")
    ax1.set_title("Cherenkov yield per particle")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Particle energy (kinetic ≈ total) [GeV]")
    ax2.set_ylabel("σ / μ  (N_total)")
    ax2.set_title("Shower-to-shower fluctuations")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_ylim(0, None)

    fig.suptitle("Geant4 Cherenkov yield — total photon count per primary particle",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "g4_yield_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Composite hadronic shower profile
# ─────────────────────────────────────────────────────────────────────────────

def plot_composite_shower(library):
    """
    For each Pythia DIS event, sum the mean G4 Cherenkov profiles of all
    stored secondaries (top20 by energy, non-zero entries only).
    Each secondary's profile is looked up at the nearest G4 grid energy in
    log-E space for its species.  Averaging over events gives the mean ± 1σ
    composite hadronic shower profile — what the detector actually sees from
    a full DIS hadronic blob.

    One panel per E_nu group in the Pythia file.
    """
    if not os.path.exists(PYTHIA_FILE):
        print(f"Skipping composite shower plot: {PYTHIA_FILE} not found.")
        return

    # Build lookup: (pid, E_GeV) → mean profile at nearest grid energy
    def nearest_mean_profile(pid, E_GeV):
        resolved = PID_PROXY.get(pid, pid)
        if resolved not in library or E_GeV <= 0:
            return None
        energies = sorted(library[resolved].keys())
        log_E    = np.log10(E_GeV)
        nearest  = min(energies, key=lambda e: abs(np.log10(e) - log_E))
        return library[resolved][nearest]["profiles"].mean(axis=0)

    # Reference z-axis from first available library entry
    ref_pid  = next(iter(library))
    ref_E    = sorted(library[ref_pid].keys())[0]
    z_edges  = library[ref_pid][ref_E]["z_edges"]
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_bins   = len(z_centers)

    with h5py.File(PYTHIA_FILE, "r") as pf:
        groups = sorted(pf.keys())
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), sharey=False)
        if n_groups == 1:
            axes = [axes]

        for ax, grp_name in zip(axes, groups):
            grp        = pf[grp_name]
            top20_e    = grp["top20_energies"][:]   # (n_events, 20) — total energy p.e()
            top20_pids = grp["top20_pids"][:]       # (n_events, 20)
            E_had      = grp["E_had"][:]
            valid      = np.where(E_had > E_HAD_MIN_GEV)[0]
            top20_e    = top20_e[valid]
            top20_pids = top20_pids[valid]
            n_events   = len(valid)

            composite = np.zeros((n_events, n_bins))

            for i in range(n_events):
                for k in range(top20_e.shape[1]):
                    pid = int(top20_pids[i, k])
                    E_k = float(top20_e[i, k])
                    if pid == 0 or E_k <= 0:
                        continue   # padding entry
                    prof = nearest_mean_profile(pid, E_k)
                    if prof is None:
                        continue   # species not in library
                    composite[i] += prof

            mean_p = composite.mean(axis=0)
            std_p  = composite.std(axis=0)

            ax.plot(z_centers, mean_p, color="steelblue", linewidth=1.5, label="mean")
            ax.fill_between(z_centers,
                            np.maximum(mean_p - std_p, 0),
                            mean_p + std_p,
                            alpha=0.3, color="steelblue", label="±1σ")
            ax.set_title(f"E_ν = {nice_label(grp_name)}")
            ax.set_xlabel("Depth in ice [cm]")
            ax.set_xlim(0, None)
            ax.legend(fontsize=8)

        axes[0].set_ylabel("Cherenkov photons / 5 cm")
        fig.suptitle(
            "Composite hadronic shower Cherenkov profile\n"
            "(G4 sub-shower profiles summed over Pythia DIS secondaries, mean ± 1σ)",
            fontsize=10)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, "g4_composite_shower_profile.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers: sampling & rescaling individual runs
# ─────────────────────────────────────────────────────────────────────────────

# Radiation length in ice (matches DISFromSpline): X0 = 36.08 g/cm^2 / rho_ice
RHO_ICE = 0.9216
X0_CM   = 36.08 / RHO_ICE   # ~39.1 cm


def _z_centers(library):
    """Reference longitudinal bin centers (assumed common to all runs)."""
    ref_pid = next(iter(library))
    ref_E   = sorted(library[ref_pid].keys())[0]
    z_edges = library[ref_pid][ref_E]["z_edges"]
    return 0.5 * (z_edges[:-1] + z_edges[1:])


def _shift_profile(prof, z_centers, dz):
    """Shift a profile by dz cm along the axis (positive = deeper), zero-filled."""
    if abs(dz) < 1e-9:
        return prof
    return np.interp(z_centers - dz, z_centers, prof, left=0.0, right=0.0)


def _nearest_grid_energy(resolved_pid, E_true, library):
    energies = sorted(library[resolved_pid].keys())
    return min(energies, key=lambda e: abs(np.log10(e) - np.log10(E_true)))


def sample_rescaled_run(pid, E_true, library, interpolators, rng,
                        z_centers, do_shift=True, return_meta=False):
    """
    Pick ONE random G4 run for (species, nearest grid energy in log-E), then
    rescale it to the true secondary energy:
      • amplitude ×= N(E_true)/N(E_grid)          (mean-yield curve, log-log)
      • depth shifted by ln(E_true/E_grid)·X0     (shower-max drift)
    Keeps the individual run's shape/multimodality. Returns photons/bin array,
    or None if the species is not in the G4 library.
    """
    if pid == 0 or E_true <= 0:
        return (None, None) if return_meta else None
    resolved = PID_PROXY.get(pid, pid)
    if resolved not in library or resolved not in interpolators:
        return (None, None) if return_meta else None
    E_grid  = _nearest_grid_energy(resolved, E_true, library)
    profs   = library[resolved][E_grid]["profiles"]
    run_idx = int(rng.integers(len(profs)))
    prof    = profs[run_idx].astype(float).copy()

    interp = interpolators[resolved]
    N_true = 10 ** interp(np.log10(E_true))
    N_grid = 10 ** interp(np.log10(E_grid))
    if N_grid > 0:
        prof *= (N_true / N_grid)

    if do_shift:
        dz = np.log(E_true / E_grid) * X0_CM
        prof = _shift_profile(prof, z_centers, dz)
    if return_meta:
        return prof, {"pid": resolved, "E_grid": E_grid, "run": run_idx}
    return prof


def make_remainder_pi0(Y_remainder, library, interpolators, rng):
    """
    Represent the un-tracked soft secondaries as a single pi0 at the origin
    carrying total yield Y_remainder. Energy is chosen by inverting the pi0
    yield curve (so the shape is physically sized), then the amplitude is
    normalized so the integral equals Y_remainder exactly. Placed at origin
    (no depth shift). Returns photons/bin array or None.
    """
    if Y_remainder <= 0 or 111 not in library or 111 not in interpolators:
        return None
    energies = sorted(library[111].keys())
    logE = np.log10(energies)
    logN = np.array([interpolators[111](le) for le in logE])
    # invert the log-log yield curve (logN increases monotonically with logE)
    logE_art = np.interp(np.log10(Y_remainder), logN, logE)
    E_grid   = min(energies, key=lambda e: abs(np.log10(e) - logE_art))
    profs    = library[111][E_grid]["profiles"]
    prof     = profs[rng.integers(len(profs))].astype(float).copy()
    cur = prof.sum()
    if cur > 0:
        prof *= (Y_remainder / cur)
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: individual runs of one (species, energy) — illustrate multimodality
# ─────────────────────────────────────────────────────────────────────────────

def _count_significant_peaks(prof, rel_prom=0.12, min_dist_bins=12, smooth=5):
    """Count prominent peaks in a (noisy) profile after light smoothing."""
    m = prof.max()
    if m <= 0:
        return 0
    if smooth > 1:
        p = np.convolve(prof, np.ones(smooth) / smooth, mode="same")
    else:
        p = prof
    peaks, _ = find_peaks(p, prominence=rel_prom * m, distance=min_dist_bins)
    return len(peaks)


def plot_individual_runs(library, pid=211, E_GeV=1000.0, n_show=10,
                         min_multimodal=3, seed=10):
    """
    10 individual G4 runs of a single (species, energy), overlaid on one axis,
    to show run-to-run fluctuation and multimodality. Selection: 10 random runs;
    if fewer than `min_multimodal` are multimodal (>=2 prominent peaks), swap
    non-multimodal picks for multimodal ones until at least `min_multimodal` are.
    """
    if pid not in library or E_GeV not in library[pid]:
        print(f"  [skip] no library entry for pid={pid}, E={E_GeV} GeV")
        return

    entry     = library[pid][E_GeV]
    profiles  = entry["profiles"].astype(float)
    z_edges   = entry["z_edges"]
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    n_events  = len(profiles)
    rng       = np.random.default_rng(seed)

    is_multi = {i: _count_significant_peaks(profiles[i]) >= 2 for i in range(n_events)}

    selected = list(rng.choice(n_events, size=min(n_show, n_events), replace=False))
    n_multi  = sum(is_multi[i] for i in selected)

    if n_multi < min_multimodal:
        pool = [i for i in range(n_events) if is_multi[i] and i not in selected]
        rng.shuffle(pool)
        for pos in [p for p in range(len(selected)) if not is_multi[selected[p]]]:
            if n_multi >= min_multimodal or not pool:
                break
            selected[pos] = pool.pop()
            n_multi += 1

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = cm.viridis
    for j, idx in enumerate(selected):
        tag = " *" if is_multi[idx] else ""
        ax.plot(z_centers, profiles[idx], lw=0.8,
                color=cmap(j / max(len(selected) - 1, 1)),
                label=f"run {idx}{tag}")

    label = PID_TO_META.get(pid, (None, str(pid), None))[1]
    e_lbl = f"{E_GeV/1000:.0f} TeV" if E_GeV >= 1000 else f"{E_GeV:.0f} GeV"
    ax.set_xlabel("Depth in ice [cm]")
    ax.set_ylabel("Cherenkov photons / 5 cm")
    ax.set_title(f"Individual G4 runs — {label} at {e_lbl}   "
                 f"({n_multi}/{len(selected)} multimodal, marked *)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "g4_individual_runs_pip_1TeV.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: sampled composite showers (nearest-E run + rescale, per event)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sampled_composite(library, interpolators, e_nu_group="E_nu_1e+04",
                           n_events_show=10, n_track=10, seed=7):
    """
    10 sampled composite hadronic showers at E_nu = 10 TeV, overlaid.
    For each Pythia event: take the top `n_track` secondaries by energy, sample
    one G4 run each at the nearest grid energy and rescale (amplitude + peak),
    sum them, then add a single artificial pi0 at the origin carrying the total
    yield of the remaining (un-tracked) secondaries.
    """
    if not os.path.exists(PYTHIA_FILE):
        print(f"  [skip] {PYTHIA_FILE} not found.")
        return

    z_centers = _z_centers(library)
    n_bins    = len(z_centers)
    rng       = np.random.default_rng(seed)

    with h5py.File(PYTHIA_FILE, "r") as pf:
        if e_nu_group not in pf:
            print(f"  [skip] group {e_nu_group} not in Pythia file "
                  f"(have {sorted(pf.keys())}).")
            return
        grp     = pf[e_nu_group]
        top_e   = grp["top20_energies"][:]   # (N, 20)
        top_pid = grp["top20_pids"][:]        # (N, 20)
        E_had   = grp["E_had"][:]             # (N,)

    valid = np.where(E_had > E_HAD_MIN_GEV)[0]
    if len(valid) == 0:
        print("  [skip] no DIS-valid events after E_had cut.")
        return
    ev_idx = rng.choice(valid, size=min(n_events_show, len(valid)), replace=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = cm.plasma
    rem_fracs = []

    for j, ev in enumerate(ev_idx):
        composite = np.zeros(n_bins)
        for k in range(min(n_track, top_e.shape[1])):
            prof = sample_rescaled_run(int(top_pid[ev, k]), float(top_e[ev, k]),
                                       library, interpolators, rng, z_centers,
                                       do_shift=True)
            if prof is not None:
                composite += prof

        Y_rem = sum(lookup_yield(int(top_pid[ev, k]), float(top_e[ev, k]),
                                 interpolators)
                    for k in range(n_track, top_e.shape[1]))
        rem = make_remainder_pi0(Y_rem, library, interpolators, rng)
        if rem is not None:
            composite += rem

        tot = composite.sum()
        if tot > 0 and rem is not None:
            rem_fracs.append(rem.sum() / tot)

        ax.plot(z_centers, composite, lw=1.4,
                color=cmap(0.1 + 0.8 * j / max(len(ev_idx) - 1, 1)),
                label=f"event {ev}")

    if rem_fracs:
        print(f"  mean artificial-pi0 (remainder) fraction over shown events: "
              f"{100*np.mean(rem_fracs):.1f}%")

    ax.set_xlabel("Depth in ice [cm]")
    ax.set_ylabel("Cherenkov photons / 5 cm")
    ax.set_title("Sampled composite hadronic showers — E_ν = 10 TeV")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "g4_sampled_composite_10TeV.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7: per-event sub-shower breakdown (replays a seeded composite)
# ─────────────────────────────────────────────────────────────────────────────

def plot_event_subshowers(library, interpolators, events=(6245, 5780),
                          e_nu_group="E_nu_1e+04", n_events_show=10,
                          n_track=10, seed=7):
    """
    Replays the seed-`seed` sampling of plot_sampled_composite (identical RNG
    sequence, same E_had cut) and, for the requested `events`, plots every
    sub-shower (the rescaled G4 run used for each hadron) plus the total
    composite — one subplot per event. Because the draws are replayed in the
    exact same order, the sub-showers are precisely those that built the
    plot_sampled_composite figure at the same seed.

    NOTE: keep n_events_show / n_track / seed equal to whatever produced the
    figure you are dissecting, or the RNG stream (and hence the runs) will differ.
    """
    if not os.path.exists(PYTHIA_FILE):
        print(f"  [skip] {PYTHIA_FILE} not found.")
        return

    z_centers = _z_centers(library)
    n_bins    = len(z_centers)
    rng       = np.random.default_rng(seed)

    with h5py.File(PYTHIA_FILE, "r") as pf:
        if e_nu_group not in pf:
            print(f"  [skip] group {e_nu_group} not in Pythia file.")
            return
        grp     = pf[e_nu_group]
        top_e   = grp["top20_energies"][:]
        top_pid = grp["top20_pids"][:]
        E_had   = grp["E_had"][:]

    valid = np.where(E_had > E_HAD_MIN_GEV)[0]
    if len(valid) == 0:
        print("  [skip] no DIS-valid events after E_had cut.")
        return
    ev_idx = rng.choice(valid, size=min(n_events_show, len(valid)), replace=False)

    wanted   = set(events)
    captured = {}   # ev -> {"subs": [(label, prof)], "composite": arr}

    # Replay the identical sampling loop over ALL selected events (so the RNG
    # state matches when we reach the wanted ones); capture the breakdown.
    for ev in ev_idx:
        composite = np.zeros(n_bins)
        subs = []
        for k in range(min(n_track, top_e.shape[1])):
            pid = int(top_pid[ev, k]); E = float(top_e[ev, k])
            prof, meta = sample_rescaled_run(pid, E, library, interpolators, rng,
                                             z_centers, do_shift=True,
                                             return_meta=True)
            if prof is not None:
                composite += prof
                lbl   = PID_TO_META.get(meta["pid"], (None, str(meta["pid"]), None))[1]
                e_txt = f"{E/1000:.1f} TeV" if E >= 1000 else f"{E:.0f} GeV"
                subs.append((f"{lbl} {e_txt}  (grid {meta['E_grid']:.0f} GeV, run {meta['run']})",
                             prof))
        Y_rem = sum(lookup_yield(int(top_pid[ev, k]), float(top_e[ev, k]), interpolators)
                    for k in range(n_track, top_e.shape[1]))
        rem = make_remainder_pi0(Y_rem, library, interpolators, rng)
        if rem is not None:
            composite += rem
            subs.append(("remainder pi0 (origin)", rem))
        if int(ev) in wanted:
            captured[int(ev)] = {"subs": subs, "composite": composite}

    missing = wanted - set(captured.keys())
    if missing:
        print(f"  [warn] requested events not in seed-{seed} selection: "
              f"{sorted(missing)}  (selected: {sorted(int(e) for e in ev_idx)})")
    show = [e for e in events if e in captured]
    if not show:
        print("  [skip] none of the requested events were selected at this seed.")
        return

    fig, axes = plt.subplots(len(show), 1, figsize=(11, 5.5 * len(show)))
    if len(show) == 1:
        axes = [axes]
    for ax, ev in zip(axes, show):
        data  = captured[ev]
        subs  = data["subs"]
        n_sub = len(subs)
        for i, (lbl, prof) in enumerate(subs):
            style = "--" if lbl.startswith("remainder") else "-"
            ax.plot(z_centers, prof, style, lw=1.1,
                    color=cm.turbo(i / max(n_sub - 1, 1)), label=lbl)
        ax.plot(z_centers, data["composite"], color="black", lw=2.4,
                label="composite (total)")
        ax.set_title(f"Event {ev} — sub-shower breakdown + composite "
                     f"(E_ν = 10 TeV, seed {seed})")
        ax.set_xlabel("Depth in ice [cm]")
        ax.set_ylabel("Cherenkov photons / 5 cm")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "g4_event_subshowers_10TeV.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    library       = load_g4_library()
    interpolators = build_interpolators(library)

    if PLOT_PROFILE_LIBRARY:
        print("\n── Plot 1: Profile library ──────────────────────────────")
        plot_profile_library(library)

    if PLOT_CUMULATIVE_K:
        print("\n── Plot 2: Cumulative k (G4 yields + Pythia DIS) ────────")
        plot_cumulative_k_g4(interpolators)

    if PLOT_YIELD_CURVES:
        print("\n── Plot 3: Yield curves per species ─────────────────────")
        plot_yield_curves(library)

    if PLOT_COMPOSITE_SHOWER:
        print("\n── Plot 4: Composite hadronic shower profile ────────────")
        plot_composite_shower(library)

    if PLOT_INDIVIDUAL_RUNS:
        print("\n── Plot 5: Individual 1 TeV pi+ runs ────────────────────")
        plot_individual_runs(library)

    if PLOT_SAMPLED_COMPOSITE:
        print("\n── Plot 6: Sampled composite showers (nearest-E + rescale) ──")
        plot_sampled_composite(library, interpolators)

    if PLOT_EVENT_SUBSHOWERS:
        print("\n── Plot 7: Per-event sub-shower breakdown ───────────────")
        plot_event_subshowers(library, interpolators)

    print("\nDone. All plots in", OUT_DIR)
