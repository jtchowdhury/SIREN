#!/usr/bin/env python3
r"""
validate_gamma_dis.py  --  validation + diagnostic plots for the gamma_subshowers
produced by GammaDISFromSpline (or the BlobDeveloper mock), in the L2/KS plot style.

Two checks the meeting asked for:

  (1) energy conservation -- per interaction, the gamma components grouped by
      parent should reconstruct each particle's sampled yield, and the particle
      energies should sum to E_had (up to the model's yield<->energy mapping). We
      plot the per-interaction total amplitude vs E_had and report the spread.

  (2) input vs output sampling match -- the alpha / beta / log10(amplitude)
      distributions coming OUT of the stored events should match the shower model
      sampled directly (the ground truth the C++ must reproduce). We overlay the
      two histograms and print the two-sample KS statistic and an L2 distance of
      the normalized histograms, for each parameter.

Input formats handled:
  * indexed interaction_parameters  (real GammaDISFromSpline output):
        subshower_count, subshower_<i>_alpha/_beta/_amplitude/_parent_pid/_parent_inst
  * awkward 'gamma_subshowers' record (BlobDeveloper / test_gamma_subshowers_local.py):
        subshower_alpha / subshower_beta / subshower_amplitude / subshower_parent

Usage:
    python validate_gamma_dis.py test_local.hdf5
    python validate_gamma_dis.py output/IceCube.hdf5 --model <shower_model.npz> --out plots/
"""
import os
import sys
import argparse
import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import ks_2samp

# ---- plot style (matches the L2/KS comparison plots) ----------------------
FONTSIZE = 20
plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE - 2,
    "ytick.labelsize": FONTSIZE - 2,
    "legend.fontsize": FONTSIZE - 3,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
C_OUT = "#1f77b4"   # stored events  (C++ / mock output)
C_REF = "#d62728"   # model ground truth
ALPHA = 0.45


def _finish(ax, xlabel, ylabel, title=None, logy=False, sci=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if sci:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(fmt)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(FONTSIZE - 3)


# ---------------------------------------------------------------------------
#  readers -> per-interaction list of dict(parent_key -> [(alpha,beta,amp),...])
# ---------------------------------------------------------------------------
def _read_awkward(path):
    """Read the gamma_subshowers record. Handles both the real column layout
    (subshower_parent_pid + subshower_parent_inst, written by SaveEvents from
    GammaDISFromSpline) and the mock layout (subshower_parent string tag)."""
    import awkward as ak
    with h5py.File(path, "r") as f:
        grp = f["Events"]
        arr = ak.from_buffers(
            ak.forms.from_json(grp.attrs["form"]), grp.attrs["length"],
            {k: np.asarray(v) for k, v in grp.items()})
    gs = arr["gamma_subshowers"]
    fields = gs.fields
    has_pid = "subshower_parent_pid" in fields
    ev_a   = ak.to_list(gs["subshower_alpha"])
    ev_b   = ak.to_list(gs["subshower_beta"])
    ev_amp = ak.to_list(gs["subshower_amplitude"])
    if has_pid:
        ev_pid  = ak.to_list(gs["subshower_parent_pid"])
        ev_inst = ak.to_list(gs["subshower_parent_inst"])
        ev_key = [[list(zip(pids, insts)) for pids, insts in zip(e_pid, e_inst)]
                  for e_pid, e_inst in zip(ev_pid, ev_inst)]
    else:
        ev_key = ak.to_list(gs["subshower_parent"])
    interactions = []
    for keys, a, b, amp in zip(ev_key, ev_a, ev_b, ev_amp):
        for k_i, a_i, b_i, amp_i in zip(keys, a, b, amp):
            groups = {}
            for key, aa, bb, mm in zip(k_i, a_i, b_i, amp_i):
                groups.setdefault(tuple(key) if has_pid else key, []).append((aa, bb, mm))
            interactions.append(groups)
    return interactions


def _read_indexed(path):
    """Read indexed interaction_parameters. Assumes a flat per-interaction table
    with columns subshower_count and subshower_<i>_*. Adapt the group path below
    to wherever SaveEvents writes interaction_parameters in your HDF5."""
    import awkward as ak
    with h5py.File(path, "r") as f:
        grp = f["Events"]
        arr = ak.from_buffers(
            ak.forms.from_json(grp.attrs["form"]), grp.attrs["length"],
            {k: np.asarray(v) for k, v in grp.items()})
    if "interaction_parameters" not in arr.fields:
        raise KeyError("no interaction_parameters field for indexed read")
    ip = arr["interaction_parameters"]
    interactions = []
    for ev in ak.to_list(ip):
        for rec in ev:
            n = int(rec.get("subshower_count", 0))
            groups = {}
            for i in range(n):
                p = f"subshower_{i}"
                key = (int(rec[f"{p}_parent_pid"]), int(rec[f"{p}_parent_inst"]))
                groups.setdefault(key, []).append(
                    (rec[f"{p}_alpha"], rec[f"{p}_beta"], rec[f"{p}_amplitude"]))
            interactions.append(groups)
    return interactions


def read_events(path):
    try:
        return _read_indexed(path)
    except Exception:
        return _read_awkward(path)


# ---------------------------------------------------------------------------
def flat_params(interactions):
    a, b, amp = [], [], []
    for groups in interactions:
        for comps in groups.values():
            for aa, bb, mm in comps:
                a.append(aa); b.append(bb); amp.append(mm)
    return np.array(a, float), np.array(b, float), np.array(amp, float)


def reference_sample(model_path, n):
    """Ground-truth parameter population: sample the shower model directly across
    a log-uniform energy range and the trained species."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    import hadronic_showers as hs
    model = hs.load_model(model_path)
    rng = np.random.default_rng(0)
    pids = [211, -211, 321, -321, 111, 130, 310, 2112, 2212]
    pids = [p for p in pids if model.has(p)]
    a, b, amp = [], [], []
    for _ in range(n):
        pid = pids[rng.integers(len(pids))]
        lo, hi = model.energy_range(pid)
        E = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
        sh = model.sample(pid, E, rng)
        a += list(sh.alpha); b += list(sh.beta); amp += list(sh.N * sh.weights)
    return np.array(a), np.array(b), np.array(amp)


def l2_hist(x, y, bins):
    hx, edges = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    w = np.diff(edges)
    return float(np.sqrt(np.sum((hx - hy) ** 2 * w)))


def overlay(ax, out, ref, xlabel, bins, logx=False):
    if logx:
        # robust range: 1st-99th percentile of the pooled positive values, so a
        # few extreme components don't collapse the histogram into one bin
        pool = np.concatenate([out[out > 0], ref[ref > 0]])
        lo, hi = np.percentile(pool, [1, 99])
        if not (hi > lo):
            lo, hi = pool.min(), pool.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), bins)
        ax.set_xscale("log")
    ax.hist(out, bins=bins, density=True, color=C_OUT, alpha=ALPHA, label="stored events")
    ax.hist(ref, bins=bins, density=True, color=C_REF, alpha=ALPHA, label="model (truth)")
    D = ks_2samp(out, ref).statistic
    L2 = l2_hist(out, ref, bins)
    ax.legend(title=rf"$D_{{\mathrm{{KS}}}}={D:.3f}$,  $L_2={L2:.3f}$")
    _finish(ax, xlabel, "density")
    return D, L2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hdf5")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(__file__), "..", "showers", "GammaShowerModel-v1.0", "shower_model.npz"))
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    interactions = read_events(args.hdf5)
    a, b, amp = flat_params(interactions)
    print(f"interactions: {len(interactions)}   gamma components: {len(a)}")

    ra, rb, ramp = reference_sample(args.model, max(len(a), 20000))

    # (2) input vs output distributions ------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    dka, l2a = overlay(ax[0], a, ra, r"$\alpha$", 60)
    dkb, l2b = overlay(ax[1], b, rb, r"$\beta$  [cm$^{-1}$]", 60)
    dkn, l2n = overlay(ax[2], amp[amp > 0], ramp[ramp > 0],
                       r"component amplitude  [photons]", 60, logx=True)
    fig.suptitle("Sampled gamma-component parameters: stored events vs model", y=1.02)
    fig.tight_layout()
    p1 = os.path.join(args.out, "gamma_dis_param_match.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")

    # (1) energy / yield conservation --------------------------------------
    tot = np.array([sum(mm for comps in g.values() for _, _, mm in comps)
                    for g in interactions], float)
    npart = np.array([len(g) for g in interactions], int)
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
    ax2[0].hist(np.log10(tot[tot > 0]), bins=50, color=C_OUT, alpha=0.7)
    _finish(ax2[0], r"$\log_{10}\,\sum_i N_i$  [photons/interaction]", "interactions")
    ax2[1].hist(npart, bins=range(0, npart.max() + 2), color=C_OUT, alpha=0.7)
    _finish(ax2[1], "parent particles / interaction", "interactions")
    fig2.tight_layout()
    p2 = os.path.join(args.out, "gamma_dis_energy.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")

    print("\n-- input vs output sampling match (lower is better) --")
    print(f"  alpha      KS D={dka:.4f}   L2={l2a:.4f}")
    print(f"  beta       KS D={dkb:.4f}   L2={l2b:.4f}")
    print(f"  amplitude  KS D={dkn:.4f}   L2={l2n:.4f}")
    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
