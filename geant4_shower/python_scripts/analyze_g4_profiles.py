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
                         NOTE: the E_nu = 100 TeV Pythia group is intentionally
                         left blank — the highest G4 point is 30 TeV, so
                         extrapolation to 100 TeV sub-cascades is unreliable.
                         Re-run after generating 100 TeV G4 samples.

  3. Yield curves      — mean total Cherenkov photon count (N_total) vs energy
                         per species (log-log). The G4 counterpart to the Pythia
                         multiplicity plot — shows which species contributes most.

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
]
PID_TO_META = {pid: (name, label, color) for pid, name, label, color in SPECIES}

# PDG IDs present in Pythia top20 but not in our G4 library → proxy mapping
PID_PROXY = {
    310:  321,   # K_S  → K⁺  (similar hadronic shower)
    130:  321,   # K_L  → K⁺
    2212: 211,   # p    → π⁺  (rough; hadronic shower, similar visible fraction)
    2112: 211,   # n    → π⁺  (rough; neutron has less direct Cherenkov)
    -2212: 211,
    -2112: 211,
}


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
    Only interpolates within the simulated energy range (no extrapolation).
    Returns dict: {pid: {"interp": callable, "E_min": float, "E_max": float}}
    """
    interpolators = {}
    for pid, e_data in library.items():
        energies = sorted(e_data.keys())
        means    = [e_data[E]["N_total"].mean() for E in energies]
        log_e    = np.log10(energies)
        log_n    = np.log10(means)
        interp   = interp1d(log_e, log_n, kind="linear",
                            fill_value=np.nan, bounds_error=False)
        interpolators[pid] = {
            "interp": interp,
            "E_min":  energies[0],
            "E_max":  energies[-1],
        }
    return interpolators


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Profile library
# ─────────────────────────────────────────────────────────────────────────────

def plot_profile_library(library):
    """
    One panel per species. Each panel overlays mean ± 1σ profiles at all
    simulated energies, color-coded by energy (log scale).
    """
    n_species = len(SPECIES)
    fig, axes = plt.subplots(1, n_species, figsize=(5 * n_species, 4), sharey=False)

    for ax, (pid, name, label, base_color) in zip(axes, SPECIES):
        if pid not in library:
            ax.set_title(f"{label}  (no data)")
            continue

        e_data    = library[pid]
        energies  = sorted(e_data.keys())
        log_e_min = np.log10(energies[0])
        log_e_max = np.log10(energies[-1])
        cmap      = cm.plasma

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
            ax.fill_between(z_centers, mean_p - std_p, mean_p + std_p,
                            alpha=0.15, color=color)

        ax.set_xlabel("Depth in ice [cm]")
        ax.set_ylabel("Cherenkov photons / 5 cm")
        ax.set_title(f"{label}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlim(0, None)

    fig.suptitle("Geant4 Cherenkov longitudinal profiles  (mean ± 1σ)", fontsize=12)
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
    Return interpolated mean N_total for (pid, E_GeV).
    Returns np.nan if E is outside the simulated range (no extrapolation).
    Returns 0 for padding entries (pid==0 or E<=0).
    Falls back to proxy pid if species not in library.
    """
    if pid == 0 or E_GeV <= 0:
        return 0.0
    resolved = PID_PROXY.get(pid, pid)
    if resolved not in interpolators:
        return 0.0
    meta  = interpolators[resolved]
    if E_GeV > meta["E_max"] or E_GeV < meta["E_min"]:
        return np.nan   # outside simulated range — caller handles this
    val = meta["interp"](np.log10(E_GeV))
    return float(10 ** val) if not np.isnan(val) else np.nan


def plot_cumulative_k_g4(interpolators):
    """
    Recreates the cumulative Cherenkov fraction vs k plot using G4-simulated
    N_total values. E_nu groups in SKIP_PYTHIA_GROUPS are left as blank panels
    (no reliable G4 coverage for their sub-cascade energies).
    """
    if not os.path.exists(PYTHIA_FILE):
        print(f"Skipping cumulative-k plot: {PYTHIA_FILE} not found.")
        return

    def nice_label(grp_name):
        e = float(grp_name.replace("E_nu_", ""))
        if e >= 1e6: return f"{e/1e6:.0f} PeV"
        if e >= 1e3: return f"{e/1e3:.0f} TeV"
        return f"{e:.0f} GeV"

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

            # Leave this panel blank — G4 coverage doesn't reach 100 TeV
            if grp_name in SKIP_PYTHIA_GROUPS:
                ax.text(0.5, 0.5,
                        "G4 data not yet available\n(max coverage: 30 TeV)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
                continue

            grp        = pf[grp_name]
            top20_e    = grp["top20_energies"][:]   # (nevents, 20)
            top20_pids = grp["top20_pids"][:]       # (nevents, 20)
            n_events, k_max = top20_e.shape

            # Look up G4 yield for each sub-cascade; drop events with any nan
            # (sub-cascade energy outside G4 range)
            g4_yields = np.array([
                [lookup_yield(int(top20_pids[i, k]), top20_e[i, k], interpolators)
                 for k in range(k_max)]
                for i in range(n_events)
            ], dtype=float)

            valid = ~np.any(np.isnan(g4_yields), axis=1)
            n_valid = valid.sum()
            if n_valid == 0:
                ax.text(0.5, 0.5, "No valid events", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue
            if n_valid < n_events:
                print(f"  {grp_name}: dropped {n_events - n_valid} events with "
                      f"sub-cascade E outside G4 range")

            g4_yields = g4_yields[valid]
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
            if n_valid < n_events:
                ax.text(0.98, 0.02, f"n={n_valid}/{n_events}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="gray")

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
    ax1.set_xlabel("Kinetic energy [GeV]")
    ax1.set_ylabel("Mean N_total  (Cherenkov photons, 300–600 nm)")
    ax1.set_title("Cherenkov yield per particle")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("Kinetic energy [GeV]")
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    library       = load_g4_library()
    interpolators = build_interpolators(library)

    print("\n── Plot 1: Profile library ──────────────────────────────")
    plot_profile_library(library)

    print("\n── Plot 2: Cumulative k (G4 yields + Pythia DIS) ────────")
    plot_cumulative_k_g4(interpolators)

    print("\n── Plot 3: Yield curves per species ─────────────────────")
    plot_yield_curves(library)

    print("\nDone. All plots in", OUT_DIR)
