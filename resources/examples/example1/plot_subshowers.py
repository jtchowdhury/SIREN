import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.special import gammaincinv

def compute_subshower_positions(E_had, E_sub=1.0):
    """Mirrors the C++ ComputeSubshowerPositions exactly."""
    N = max(1, int(round(E_had / E_sub)))
    E_c = 0.2
    alpha = 0.3 + 0.7 * np.log(E_had / E_c)
    beta  = 0.9
    rho_ice = 0.9216
    X0_cm = 36.08 / rho_ice  # ~39.1 cm

    r = (2*np.arange(1, N+1) - 1) / (2*N)
    x_rad = gammaincinv(alpha, r) / beta   # same as gamma_p_inv in boost
    positions_cm = x_rad * X0_cm
    return positions_cm, alpha, beta, X0_cm

def subshower_profile(L, x_i, alpha_sub=3.0, beta_sub=2.0, X0_cm=39.1):
    """Single 1 GeV EM sub-shower profile centered at x_i."""
    t = (L - x_i) / X0_cm   # convert to radiation lengths
    t = np.where(t > 0, t, 0)
    return gamma.pdf(t, a=alpha_sub, scale=1.0/beta_sub) / X0_cm

# --- Plot for three different hadronic energies ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, E_had in zip(axes, [10, 100, 1000]):  # GeV
    positions_cm, alpha, beta, X0_cm = compute_subshower_positions(E_had, E_sub=1.0)
    N = len(positions_cm)

    # x-axis: distance along shower axis in meters
    L = np.linspace(0, max(positions_cm)*1.5, 1000)

    # sum of all sub-shower profiles
    total_profile = np.zeros_like(L)
    for x_i in positions_cm:
        total_profile += subshower_profile(L, x_i)
    total_profile /= N  # normalize

    # also plot the target gamma distribution for comparison
    x_rad = L / X0_cm
    target = gamma.pdf(x_rad, a=alpha, scale=1.0/beta) / X0_cm

    ax.plot(L/100, total_profile, label=f"Sum of {N} sub-showers", color="steelblue")
    ax.plot(L/100, target, label="Target gamma dist", color="red", linestyle="--")
    ax.set_xlabel("Distance along shower axis [m]")
    ax.set_ylabel("Profile (arb. units)")
    ax.set_title(f"E_had = {E_had} GeV  (α={alpha:.1f}, β={beta:.1f})")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("output/subshower_profile_check.png", dpi=150)
print("Saved to output/subshower_profile_check.png")