import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import gammaincinv

def compute_subshower_positions(E_had, E_sub=1.0):
    """Mirrors C++ ComputeSubshowerPositions exactly."""
    N = max(1, int(round(E_had / E_sub)))
    alpha = 0.3 + 0.7 * np.log(E_had / 0.2)
    beta  = 0.9
    X0_cm = 36.08 / 0.9216  # radiation length in ice [cm]
    r = (2 * np.arange(1, N + 1) - 1) / (2 * N)
    positions_cm = gammaincinv(alpha, r) / beta * X0_cm
    return positions_cm, alpha, beta, X0_cm

BIN_WIDTH_M = 0.1  # fixed 10 cm bins throughout

def scaled_target(alpha, beta, X0_cm, max_pos_m, peak_value):
    """Gamma PDF scaled to match histogram peak."""
    X0_m = X0_cm / 100
    L_m  = np.linspace(0, max_pos_m, 1000)
    pdf  = gamma.pdf(L_m / X0_m, a=alpha, scale=1.0 / beta) / X0_m
    return L_m, pdf / pdf.max() * peak_value


# ── Plot 1: varying E_had, fixed E_sub = 1 GeV ────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, E_had in zip(axes, [100, 1000, 100000]):
    positions_m = compute_subshower_positions(E_had, E_sub=1.0)[0] / 100
    N    = len(positions_m)
    E_sub = E_had / N  # = 1.0 GeV

    bins = np.arange(0, positions_m.max() * 1.5, BIN_WIDTH_M)
    counts, edges = np.histogram(positions_m, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    energy_per_bin = counts * E_sub

    L_m, target = scaled_target(*compute_subshower_positions(E_had, 1.0)[1:],
                                 positions_m.max() * 1.5, energy_per_bin.max())

    ax.bar(centers, energy_per_bin, width=BIN_WIDTH_M,
           alpha=0.6, color="steelblue", label=f"N = {N} sub-showers")
    ax.plot(L_m, target, 'r--', label="Target shape (scaled)")
    ax.set_xlabel("Distance along shower axis [m]")
    ax.set_ylabel("Energy deposited [GeV / 10 cm]")
    ax.set_title(f"E_had = {E_had} GeV  (α={0.3+0.7*np.log(E_had/0.2):.1f}, β=0.9)")
    ax.legend(fontsize=8)

fig1.suptitle("Varying E_had, fixed E_sub = 1 GeV", fontsize=12)
fig1.tight_layout()
fig1.savefig("/n/home13/jchowdhury/SIREN/resources/analysis/output/subshower_profile_Ehad.png", dpi=150)
print("Saved output/subshower_profile_Ehad.png")


# ── Plot 2: fixed E_had = 1 TeV, varying E_sub ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

E_had  = 1000  # GeV (1 TeV)
E_subs = [1.0, 5.0, 10.0]
colors = ["steelblue", "salmon", "green"]

for E_sub, color in zip(E_subs, colors):
    positions_m, alpha, beta, X0_cm = compute_subshower_positions(E_had, E_sub)
    positions_m = positions_m / 100
    N = len(positions_m)

    bins = np.arange(0, positions_m.max() * 1.5, BIN_WIDTH_M)
    counts, edges = np.histogram(positions_m, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    ax2.bar(centers, counts, width=BIN_WIDTH_M,
            alpha=0.5, color=color, label=f"E_sub = {E_sub:.0f} GeV  (N = {N})")

# target shape scaled to N=1000 peak (E_sub=1 GeV case)
positions_1, alpha, beta, X0_cm = compute_subshower_positions(E_had, 1.0)
positions_1 /= 100
bins_1  = np.arange(0, positions_1.max() * 1.5, BIN_WIDTH_M)
counts_1, _ = np.histogram(positions_1, bins=bins_1)
L_m, target = scaled_target(alpha, beta, X0_cm, positions_1.max() * 1.5, counts_1.max())

#ax2.plot(L_m, target, 'r--', linewidth=2, label="Target shape (scaled)")
ax2.set_xlabel("Distance along shower axis [m]")
ax2.set_ylabel("# of sub-showers per 10 cm bin")
ax2.set_title(f"E_had = {E_had} GeV: effect of E_sub choice  (α={alpha:.1f}, β=0.9)")
ax2.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig("/n/home13/jchowdhury/SIREN/resources/analysis/output/subshower_profile_Esub.png", dpi=150)
print("Saved /n/home13/jchowdhury/SIREN/resources/analysis/output/subshower_profile_Esub.png")