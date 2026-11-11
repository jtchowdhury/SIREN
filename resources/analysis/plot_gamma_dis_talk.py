#!/usr/bin/env python3
"""
plot_gamma_dis_talk.py -- "spaghetti" shower plots for talks. Every plot is its
own file (no panels). Profiles are the unit-area density l^-1 dl/dx vs depth x.

Single particles (1 TeV by default, 100 showers each, e- and pi+ overlaid):
  single_g4_raw.png        Geant4 showers as stored
  single_g4_conv45.png     Geant4 showers convolved with a Gaussian (sigma = --sigma-cm)
  single_sampled.png       showers sampled from the gamma-mixture model

Neutrino events at each --enu (CC and NC overlaid, 100 events each). The SAME
events (same SIREN kinematics, same boosted PYTHIA hadron list) are used in
every plot, so only the shower description changes:
  events_E<E>_sampled.png     GammaDISFromSpline output subshowers (SIREN)
  events_E<E>_g4_raw.png      one stored G4 shower per hadron (+ the electron in
                              CC), summed; nearest library energy, light rescaled
  events_E<E>_g4_conv45.png   the same G4 showers, convolved
  events_E<E>_analytic.png    DISFromSpline: Aachen pi+ profile of the blob E_had
                              (+ Aachen e- profile of the electron in CC)

Only the G4 curves are convolved (their bin-to-bin spikes are not resolvable).
sigma = 45 cm ~ 2 ns DOM timing x c/n_ice; change with --sigma-cm.

Usage (cluster):
    python plot_gamma_dis_talk.py --samples output/gamma_dis_samples \
        --g4-dir ~/SIREN/geant4_shower/output/data/h5 --enu 1e4 1e5
"""
import os
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import gamma_dis_common as G
import plot_gamma_dis_ccnc as P          # profiles, Aachen pi+, G4 library tools
import hadronic_showers as hs

# colours of the reference figure: e- / CC blue, pi+ / NC orange
C_BLUE, C_ORANGE = "#3a87b0", "#e0861a"
# Aachen e- longitudinal profile (fennel aachen.pkl, 'em cascade' -> 11)
AACHEN_E_ALPHA, AACHEN_E_BETA, AACHEN_E_B = 2.01849, 1.45469, 0.63207


def analytic_em(x, E, yield_tot):
    a = max(AACHEN_E_ALPHA + AACHEN_E_BETA * np.log10(E), 1e-3)
    return yield_tot * hs._kernel(x / P.AACHEN_X0_CM, a, AACHEN_E_B) / P.AACHEN_X0_CM


def norm(x, y):
    area = hs._trapz(y, x)
    return y / area if area > 0 else y


def conv(y, sigma_cm, binw):
    return gaussian_filter1d(np.asarray(y, float), sigma_cm / binw, mode="constant") \
        if sigma_cm > 0 else np.asarray(y, float)


def spaghetti(x, groups, path, xlim, legend_title=None):
    """groups: list of (list of profiles, colour, label). One file per call."""
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    for profs, col, lab in groups:
        for i, y in enumerate(profs):
            ax.plot(x, norm(x, y), color=col, lw=0.8, alpha=0.55,
                    label=lab if i == 0 else None)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$x$ [cm]", fontsize=16)
    ax.set_ylabel(r"$\hat{\ell}_{\mathrm{tot}}^{-1}\mathrm{d}\hat{\ell}/\mathrm{d}x$ [1/cm]",
                  fontsize=16)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    ax.tick_params(direction="in", top=True, right=True, labelsize=14, length=5)
    ax.xaxis.get_offset_text().set_fontsize(13)
    ax.yaxis.get_offset_text().set_fontsize(13)
    leg = ax.legend(fontsize=12, title=legend_title, title_fontsize=12, framealpha=0.9)
    for lh in leg.get_lines():
        lh.set_linewidth(1.6); lh.set_alpha(1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="output/gamma_dis_samples")
    ap.add_argument("--g4-dir", nargs="+", required=True)
    ap.add_argument("--model", default=os.path.join(
        G.SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.npz"))
    ap.add_argument("--outdir", default="output/gamma_dis_plots/talk")
    ap.add_argument("--single-energy", type=float, default=1000.0)
    ap.add_argument("--enu", type=float, nargs="+", default=[1e4, 1e5])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sigma-cm", type=float, default=45.0)
    ap.add_argument("--xlim-single", type=float, nargs=2, default=[0, 1500])
    ap.add_argument("--xlim-events", type=float, nargs=2, default=[0, 2000])
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*outside the trained range.*")
    rng = np.random.default_rng(args.seed)
    tag = f"conv{args.sigma_cm:g}"
    conv_title = rf"Geant4, convolved ($\sigma$ = {args.sigma_cm:g} cm)"

    EV, _C, lib = P.load_g4_tools([os.path.expanduser(d) for d in args.g4_dir])
    any_pid = next(iter(lib)); any_E = next(iter(lib[any_pid]))
    x = np.asarray(lib[any_pid][any_E]["z_centers"], float)
    binw = float(x[1] - x[0])
    model = hs.load_model(args.model)
    out = lambda name: os.path.join(args.outdir, name)

    # ---------------- single particles ----------------
    E1 = args.single_energy
    el = G.elabel(E1)
    sp = [(11, "em", C_BLUE, rf"{el} $e^-$ ({args.n} runs)"),
          (211, "pip", C_ORANGE, rf"{el} $\pi^+$ ({args.n} runs)")]
    raw = {}
    for pid, name, _c, _l in sp:
        if pid not in lib or E1 not in lib[pid]:
            raise SystemExit(f"no G4 library file for {name} at {E1:g} GeV")
        P_ = lib[pid][E1]["profiles"]
        raw[pid] = P_[rng.choice(len(P_), min(args.n, len(P_)), replace=False)]
    spaghetti(x, [(raw[p], c, l) for p, _n, c, l in sp], out("single_g4_raw.png"),
              args.xlim_single, "Geant4")
    spaghetti(x, [([conv(y, args.sigma_cm, binw) for y in raw[p]], c, l) for p, _n, c, l in sp],
              out(f"single_g4_{tag}.png"), args.xlim_single, conv_title)
    spaghetti(x, [([model.sample(n, E1, rng, x=x).photons for _ in range(args.n)], c, l)
                  for _p, n, c, l in sp],
              out("single_sampled.png"), args.xlim_single, "Gamma-mixture model")

    # ---------------- neutrino events ----------------
    for Enu in args.enu:
        el = G.elabel(Enu)
        prof = {"sampled": {}, "g4": {}, "analytic": {}}
        for ch in ("cc", "nc"):
            path = os.path.join(args.samples, f"ccnc_mono_E{Enu:.0e}_{ch}.npz")
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}: run gamma_dis_ccnc_samples.py "
                                 f"--energies {Enu:g}")
            s = G.load_sample(path)
            idx = P.pick(rng, np.arange(len(s["E_nu"])), args.n)
            smp, g4, ana = [], [], []
            for e in idx:
                # same hadron list for everything: SIREN's boosted PYTHIA hadrons
                h0, h1 = s["had_off"][e], s["had_off"][e + 1]
                parts = [(int(s["had_pid"][i]), G.e_kin(s["had_pid"][i], float(s["had_p"][i, 0])))
                         for i in range(h0, h1)]
                if ch == "cc":
                    parts.append((11, G.e_kin(11, float(s["lep"][e, 0]))))
                missing = [pid for pid, E in parts if E > 0 and pid not in lib]
                if missing:
                    print(f"  note: {ch} event {e}: no G4 library for {missing}, left out of G4")
                parts = [(pid, E) for pid, E in parts if E > 0 and pid in lib]
                smp.append(P.output_profile(s, e, x))                  # SIREN output
                g4.append(P.g4_composite(EV, lib, parts, len(x), rng))  # stored G4 showers
                Eh = max(float(s["blob"][e, 0]) - P.M_NUCLEON, 1e-3)    # DISFromSpline blob
                a = P.analytic_had(x, Eh, P.mean_yield(model, "pip", Eh))
                if ch == "cc":
                    Ee = G.e_kin(11, float(s["lep"][e, 0]))
                    a = a + analytic_em(x, Ee, P.mean_yield(model, "em", Ee))
                ana.append(a)
            prof["sampled"][ch], prof["g4"][ch], prof["analytic"][ch] = smp, g4, ana
            if ch == "cc" and Enu > 1e4:
                print(f"  note: at E_nu = {el} the electron (~(1-y) E_nu) is above the 10 TeV "
                      "e- G4 library / model range: its shape is the 10 TeV one, rescaled.")
        lab = lambda ch: f"{el} {ch.upper()} ({args.n} events)"
        grp = lambda kind, f=lambda y: y: [([f(y) for y in prof[kind][ch]], col, lab(ch))
                                           for ch, col in (("cc", C_BLUE), ("nc", C_ORANGE))]
        stem = f"events_E{Enu:.0e}"
        spaghetti(x, grp("sampled"), out(f"{stem}_sampled.png"), args.xlim_events,
                  "GammaDISFromSpline")
        spaghetti(x, grp("g4"), out(f"{stem}_g4_raw.png"), args.xlim_events, "Geant4")
        spaghetti(x, grp("g4", lambda y: conv(y, args.sigma_cm, binw)),
                  out(f"{stem}_g4_{tag}.png"), args.xlim_events, conv_title)
        spaghetti(x, grp("analytic"), out(f"{stem}_analytic.png"), args.xlim_events,
                  "DISFromSpline + analytic profile")


if __name__ == "__main__":
    main()
