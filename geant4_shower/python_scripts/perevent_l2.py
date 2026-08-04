"""
perevent_l2.py
==============
Per-event L2 validation (what Andy asked for).

Idea: take the ACTUAL Geant4 showers (the injected events) and run each one
through the pipeline -- i.e. reconstruct it with each modeling form -- then
compare that reconstruction to the SAME G4 shower with a plain L2 norm, one
shower at a time.  Because each reconstruction comes from a specific G4 event,
there is a true 1-to-1 correspondence, so no distribution / mean / fluctuation
machinery is needed.  Only plain L2 is used.

For every G4 shower g we compute the relative L2 error

        L2(g) = sum_x (g - reconstruction)^2  /  sum_x g^2

for four ways of representing the shower:

    Analytic (fixed shape)   SIREN placeholder gamma, best amplitude only
    Single Gamma Fit         one gamma, least squares
    2-Gamma Fit              exactly two gammas, least squares
    Gamma Mixture Fit        sum of m gammas (BIC), least squares

then average over events at each energy and plot vs shower energy.  Lower =
the pipeline reproduces real showers better.

NOTE: this tests how well the pipeline can REPRESENT / reconstruct a real
shower.  It does not, on its own, test whether the random sampler reproduces
the event-to-event fluctuation (there is no per-event match for a freshly
sampled shower) -- that is a separate question.

Run
---
    python perevent_l2.py --species pip
    # data: ../output/data ; plots: output/plots/perevent_l2

Only the G4 library is needed (no model pkl) -- this is pure fitting.
"""

import os
import argparse
import numpy as np

from shower_gamma_model import load_g4_library, NAME_TO_PID, PID_TO_NAME
import compare_methods as C
import compare_fixed2gamma as F   # reuse fit_l2_forms_alt + labels


def run_species(library, pid, outdir, n_fit, rng):
    name = PID_TO_NAME.get(pid, str(pid))
    rows = []
    for E in sorted(library[pid].keys()):
        x = library[pid][E]["z_centers"]
        g4 = library[pid][E]["profiles"]
        # per-event relative L2 of each form, averaged over a subsample of events
        fl = F.fit_l2_forms_alt(g4, x, E, n_fit, rng)
        for form, val in fl.items():
            rows.append(dict(species=name, E=E, form=form, fit_l2=val))
        print(f"  E={E:8.0f}  " +
              "  ".join(f"{k}={v:.3g}" for k, v in fl.items()))

    C._write_fit_csv(rows, os.path.join(outdir, f"perevent_l2_{name}.csv"))
    sp = C.SPECIES_LATEX.get(name, name)
    C._one_plot(
        rows, "fit_l2",
        f"Per-Event Reconstruction $L_2$ vs Geant4 ({sp})",
        r"$\langle\, \sum_x (g-\mathrm{model})^2 / \sum_x g^2 \,\rangle$  per event",
        os.path.join(outdir, f"perevent_l2_{name}.png"),
        method_tags=["analytic", "single", "twogamma", "mixture"],
        meta=F.FIT_META_ALT, match_key="form", include_g4=False, logy=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g4-dir", default="../output/data")
    ap.add_argument("--species", default="pip")
    ap.add_argument("--n-fit", type=int, default=200,
                    help="G4 showers per energy to reconstruct (slow: curve-fits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir",
                    default="/n/home13/jchowdhury/SIREN/geant4_shower/output/plots/perevent_l2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    library = load_g4_library(args.g4_dir)
    rng = np.random.default_rng(args.seed)
    print(f"species in G4 dir: {sorted(PID_TO_NAME.get(p, str(p)) for p in library)}")

    for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
        pid = NAME_TO_PID.get(sp)
        if pid is None:
            print(f"skip unknown species '{sp}'"); continue
        if pid not in library:
            print(f"skip '{sp}': no shower_{sp}_E*GeV.h5 in {args.g4_dir}"); continue
        print(f"== {sp} ==")
        run_species(library, pid, args.outdir, args.n_fit, rng)


if __name__ == "__main__":
    main()
