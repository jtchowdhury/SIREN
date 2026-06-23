"""
analyze_dis_secondaries.py
--------------------------
Loads the HDF5 file produced by generate_pythia_dis.py and makes three plots:

  1. Cumulative Cherenkov fraction vs k (how many sub-cascades do you need?)
  2. EM fraction f_EM distribution (dominant source of shower-to-shower variance)
  3. Mean particle multiplicity vs E_had (what comes out of the hadronic blob?)

Run after generate_pythia_dis.py:
    python analyze_dis_secondaries.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

INFILE = "output/pythia_dis_secondaries.h5"


def cumulative_cherenkov(top20_e, top20_cw, k_max=20):
    """
    For each event compute the cumulative Cherenkov fraction when tracking
    the top k sub-cascades (sorted by energy descending).

    Returns array of shape (n_events, k_max).
    """
    yields     = top20_e * top20_cw                          # (n_events, 20)
    total      = yields.sum(axis=1, keepdims=True)           # (n_events, 1)
    # avoid divide-by-zero for padded zeros
    total      = np.where(total == 0, 1.0, total)
    cumulative = np.cumsum(yields, axis=1) / total           # (n_events, 20)
    return cumulative[:, :k_max]


def nice_label(grp_name):
    """'E_nu_1e+03' -> '1 TeV'  (rough, for plot titles)"""
    e = float(grp_name.replace("E_nu_", ""))
    if e >= 1e6:   return f"{e/1e6:.0f} PeV"
    if e >= 1e3:   return f"{e/1e3:.0f} TeV"
    return f"{e:.0f} GeV"


def plot_cumulative_cherenkov(f, groups, outdir):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    for ax, grp_name in zip(axes, groups):
        grp     = f[grp_name]
        top20_e = grp["top20_energies"][:]
        top20_cw= grp["top20_cher_weights"][:]

        cumfrac  = cumulative_cherenkov(top20_e, top20_cw, k_max=20)
        mean_cf  = cumfrac.mean(axis=0)
        std_cf   = cumfrac.std(axis=0)
        k        = np.arange(1, 21)

        # find k where mean first exceeds 0.90
        k90 = int(np.argmax(mean_cf >= 0.90)) + 1 if (mean_cf >= 0.90).any() else 20

        ax.plot(k, mean_cf, "o-", color="steelblue", ms=4, label="mean")
        ax.fill_between(k, mean_cf - std_cf, mean_cf + std_cf,
                        alpha=0.25, color="steelblue", label="±1σ")
        ax.axhline(0.90, color="r", linestyle="--", linewidth=0.9,
                   label=f"90% (k={k90})")
        ax.axvline(k90, color="r", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("# sub-cascades tracked (top k by energy)")
        ax.set_title(f"E_ν = {nice_label(grp_name)}")
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Cumulative Cherenkov fraction")
    fig.suptitle("How many sub-cascades are needed to capture 90% of Cherenkov yield?",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "cumulative_cherenkov_fraction.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_em_fraction(f, groups, outdir):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4))
    if len(groups) == 1:
        axes = [axes]

    for ax, grp_name in zip(axes, groups):
        grp  = f[grp_name]
        f_EM = grp["f_EM"][:]

        ax.hist(f_EM, bins=40, color="steelblue", alpha=0.7, density=True,
                label="events")
        ax.axvline(f_EM.mean(), color="r", linestyle="--",
                   label=f"⟨f_EM⟩ = {f_EM.mean():.3f}\nσ = {f_EM.std():.3f}")
        ax.set_xlabel("EM fraction  f_EM = E(π⁰) / E_had")
        ax.set_ylabel("Probability density")
        ax.set_title(f"E_ν = {nice_label(grp_name)}")
        ax.legend(fontsize=8)

    fig.suptitle("Event-to-event EM fraction fluctuations  (dominant Cherenkov variance)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "em_fraction_distribution.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def plot_multiplicity(f, groups, outdir):
    n_col = min(len(groups), 2)
    n_row = (len(groups) + 1) // 2
    fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 4 * n_row))
    axes = np.array(axes).flatten()

    particle_styles = [
        ("n_pi0",    "π⁰",  "steelblue"),
        ("n_pich",   "π±",  "darkorange"),
        ("n_kaon",   "K",   "green"),
        ("n_nucleon","p/n", "purple"),
    ]

    for ax, grp_name in zip(axes, groups):
        grp   = f[grp_name]
        E_had = grp["E_had"][:]

        # Log-uniform bins in E_had
        e_min = max(E_had.min(), 0.5)
        e_max = E_had.max()
        e_bins   = np.logspace(np.log10(e_min), np.log10(e_max), 25)
        e_centers= np.sqrt(e_bins[:-1] * e_bins[1:])   # geometric midpoints
        bin_idx  = np.digitize(E_had, e_bins)

        for key, name, color in particle_styles:
            counts = grp[key][:]
            means  = np.array([
                counts[bin_idx == i].mean() if (bin_idx == i).any() else np.nan
                for i in range(1, len(e_bins))
            ])
            ax.plot(e_centers, means, "o-", label=name, color=color, ms=3)

        ax.set_xscale("log")
        ax.set_xlabel("E_had [GeV]")
        ax.set_ylabel("Mean particle count per event")
        ax.set_title(f"E_ν = {nice_label(grp_name)}")
        ax.legend(fontsize=8)

    # hide any unused subplots
    for ax in axes[len(groups):]:
        ax.set_visible(False)

    fig.suptitle("Mean secondary multiplicity vs E_had  (binned from DIS events)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "multiplicity_vs_Ehad.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(INFILE):
        print(f"ERROR: {INFILE} not found. Run generate_pythia_dis.py first.")
        raise SystemExit(1)

    outdir = "output"
    os.makedirs(outdir, exist_ok=True)

    with h5py.File(INFILE, "r") as f:
        groups = sorted(f.keys())
        print(f"Found energy groups: {groups}\n")

        plot_cumulative_cherenkov(f, groups, outdir)
        plot_em_fraction(f, groups, outdir)
        plot_multiplicity(f, groups, outdir)

    print("\nAll plots saved to output/")
