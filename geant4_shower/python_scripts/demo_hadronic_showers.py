"""
demo_hadronic_showers.py
========================
Demonstrate the trained gamma-mixture shower model (no Geant4 at runtime):

  A) sample MANY showers for ONE hadron (type, energy) and overplot the
     longitudinal Cherenkov profiles -- shows the event-to-event fluctuation
     the model produces at the single-hadron level.

  B) given a list of final-state hadrons (type, energy) -- e.g. from the
     PYTHIA nu-N DIS final state -- sample one shower each, all starting at the
     common interaction vertex, and show them plus the COMPOSITE (sum) shower.

Loads the model pickle built by shower_gamma_model.py. The PYTHIA final-state
list here is a placeholder: replace EXAMPLE_FINAL_STATE or pass --final-state
<file> with a two-column "species energy_GeV" table of your generated hadrons.

This is the same call sequence you would wire into SIREN:
    interp   = load_model(pkl)
    sampler  = ShowerSampler(interp)
    prof, _  = sampler.sample_profile(pid, E, rng)      # one hadron -> one shower
    event    = sum of prof over the final-state hadrons # composite hadronic shower
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from shower_gamma_model import load_model, ShowerSampler, NAME_TO_PID, PID_TO_NAME

# --- example final-state hadrons (species_name, energy_GeV) at the vertex.
#     Replace with your PYTHIA output, or use --final-state <file>.
EXAMPLE_FINAL_STATE = [
    ("pip", 1200.0), ("pim", 800.0), ("pip", 450.0), ("p", 900.0),
    ("pim", 300.0), ("Kp", 250.0), ("n", 600.0), ("pi0", 150.0),
]

names = { 
    "pip" : r'$\pi^+$', 
    "pim" : r'$\pi^-$', 
    "pi0" : r'$\pi^0$', 
    "Kp"  : r'$K^+$', 
    "Km"  : r'$K^-$', 
    "Ks"  : r'$K^0_S$', 
    "Kl"  : r'$K^0_L$' 
}


def _known(sampler, species):
    pid = NAME_TO_PID.get(species)
    if pid is None or pid not in sampler.interp.pid_models:
        print(f"  skip '{species}': not in the model")
        return None
    return pid


def sample_many(sampler, species, E, n, seed=0):
    """n sampled profiles for one (species, energy). Returns (z_centers, profiles)."""
    pid = _known(sampler, species)
    if pid is None:
        raise SystemExit(f"species '{species}' not in the model")
    x = sampler.interp.z_centers(pid)
    rng = np.random.default_rng(seed)
    profs = np.array([sampler.sample_profile(pid, E, rng, x=x)[0] for _ in range(n)])
    return x, profs


def plot_many(x, profs, species, E, out):
    plt.figure(figsize=(8, 5))
    for p in profs:
        plt.plot(x / 100.0, p, color="#4c78a8", lw=0.5, alpha=0.5)
    plt.plot(x / 100.0, profs.mean(0), color="#c0392b", lw=1.5, label="mean")
    plt.xlabel("depth  z  [m]"); plt.ylabel("Cherenkov photons / bin")
    plt.xlim(0,17)
    plt.title(f"{len(profs)} sampled {names[species]} showers at {E/1000:.0f} TeV")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()
    print("wrote", out)


def plot_event(sampler, final_state, out_hadrons, out_composite, seed=1):
    """Sample one shower per final-state hadron; write TWO plots -- the individual
    hadron showers, and the composite (sum)."""
    rng = np.random.default_rng(seed)
    total, xref = None, None
    fig1, ax1 = plt.subplots(figsize=(8.4, 5.2))
    for sp, E in final_state:
        pid = _known(sampler, sp)
        if pid is None:
            continue
        x = sampler.interp.z_centers(pid)
        prof, info = sampler.sample_profile(pid, E, rng, x=x)
        xref = x if xref is None else xref
        total = prof.copy() if total is None else total + prof
        ax1.plot(x / 100.0, prof, lw=1.3, alpha=0.85,
                 label=f"{sp} {E:.0f} GeV (m={info['m']})")
    if total is None:
        print("  no valid hadrons to plot"); plt.close(fig1); return
    ax1.set_xlabel("depth  z  [m]"); ax1.set_ylabel("photons / bin")
    ax1.set_xlim(0,17)
    ax1.set_title("Sampled shower of each final-state hadron")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig(out_hadrons, dpi=150); plt.close(fig1)
    print("wrote", out_hadrons)

    fig2, ax2 = plt.subplots(figsize=(8.4, 5.2))
    ax2.plot(xref / 100.0, total, color="k", lw=2.6)
    ax2.set_xlabel("depth  z  [m]"); ax2.set_ylabel("photons / bin")
    ax2.set_xlim(0,17)
    ax2.set_title("Composite hadronic shower (sum)"); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(out_composite, dpi=150); plt.close(fig2)
    print("wrote", out_composite)


def load_final_state(path):
    fs = []
    for line in open(path):
        parts = line.split("#")[0].split()
        if len(parts) >= 2:
            fs.append((parts[0], float(parts[1])))
    return fs


def load_final_state_h5(path, enu_gev, event, rng):
    """One DIS event's hadronic secondaries from pythia_dis_secondaries.h5.
    Groups are 'E_nu_1e+02'..'E_nu_1e+05'; each event stores its 20 brightest
    secondaries as top20_pids / top20_energies [GeV]. Returns [(species, E), ...]
    keeping only species the model knows; K0/K0bar (311) -> KS/KL 50/50; e/mu/
    gamma/exotics are skipped (not hadrons in this model)."""
    import h5py
    key = f"E_nu_{enu_gev:.0e}"                       # e.g. 'E_nu_1e+03'
    with h5py.File(path, "r") as f:
        if key not in f:
            raise SystemExit(f"{key} not in {path}; groups = {list(f.keys())}")
        g = f[key]
        pids = np.asarray(g["top20_pids"][event])
        ens = np.asarray(g["top20_energies"][event])
    fs, skipped = [], []
    for pid, E in zip(pids, ens):
        pid = int(pid)
        if pid == 0 or not np.isfinite(E) or E <= 0:  # padding
            continue
        name = PID_TO_NAME.get(pid)
        if name is None and abs(pid) == 311:          # K0 / K0bar -> KS or KL
            name = "KS" if rng.random() < 0.5 else "KL"
        if name is None:
            skipped.append(pid); continue             # e/mu/gamma/exotic
        fs.append((name, float(E)))
    if skipped:
        print(f"  (skipped non-hadron/unknown pids: {sorted(set(skipped))})")
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../output/data/shower_model.pkl")
    ap.add_argument("--species", default="pip", help="hadron for the many-samples plot (A)")
    ap.add_argument("--energy", type=float, default=1000.0, help="its energy [GeV]")
    ap.add_argument("--n", type=int, default=100, help="how many to sample for (A)")
    ap.add_argument("--final-state", default=None,
                    help="two-column 'species energy_GeV' file for (B)")
    ap.add_argument("--final-state-h5", default=None,
                    help="pythia_dis_secondaries.h5 for (B)")
    ap.add_argument("--enu", type=float, default=1000.0,
                    help="neutrino-energy group in the h5 (100/1e3/1e4/1e5 GeV)")
    ap.add_argument("--event", type=int, default=0, help="event index in the h5 (0..9999)")
    ap.add_argument("--outdir", default="../output/plots/demo")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    interp = load_model(args.model)
    sampler = ShowerSampler(interp)

    # A) many showers for one hadron
    x, profs = sample_many(sampler, args.species, args.energy, args.n)
    plot_many(x, profs, args.species, args.energy,
              os.path.join(args.outdir, f"sample_{args.species}_{int(args.energy)}GeV.png"))

    # B) one shower per final-state hadron + composite
    if args.final_state_h5:
        fs = load_final_state_h5(args.final_state_h5, args.enu, args.event,
                                 np.random.default_rng(0))
        print(f"event {args.event} @ E_nu={args.enu:.0e} GeV -> {len(fs)} hadrons: "
              f"{[(s, round(E)) for s, E in fs]}")
    elif args.final_state:
        fs = load_final_state(args.final_state)
    else:
        fs = EXAMPLE_FINAL_STATE
    plot_event(sampler, fs,
               os.path.join(args.outdir, "event_hadrons.png"),
               os.path.join(args.outdir, "event_composite.png"))


if __name__ == "__main__":
    main()
