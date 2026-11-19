#!/usr/bin/env python3
"""
test_gamma_dis_conservation.py -- quick runtime check of GammaDISFromSpline.

Injects a few CC or NC nu_e events and checks, on the live interaction records:

  1. it runs and produces GammaSubshower secondaries;
  2. the hadron list plus the decay neutrinos and muons add up to the hadronic
     system: Sum hadron<i> + invisible + muon == had_*;
  3. the event conserves 4-momentum: lep_* + had_* == p_nu + p_target, with the
     mass of the nucleon Pythia actually used (target_pdg).

LHAPDF_DATA_PATH must point to the PDF set (HERAPDF20_NLO_EIG by default).

    export SIREN_ROOT=$HOME/SIREN
    python test_gamma_dis_conservation.py [--nc] [--n 25]
"""
import os
import argparse
import numpy as np
import siren
from siren._util import GenerateEvents

SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
MODEL = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.json")
GAMMASUB = -2000002200
NUCLEON_MASS = {2212: 0.93827208816, 2112: 0.93956542052}


def four(ip, prefix):
    return np.array([ip.get(prefix + "_E", 0.0), ip.get(prefix + "_px", 0.0),
                     ip.get(prefix + "_py", 0.0), ip.get(prefix + "_pz", 0.0)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc", action="store_true")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--emin", type=float, default=1e3)
    ap.add_argument("--emax", type=float, default=1e6)
    args = ap.parse_args()
    if not os.environ.get("LHAPDF_DATA_PATH"):
        print("WARNING: LHAPDF_DATA_PATH is not set; Pythia will not find the PDF set.")

    nue = siren.dataclasses.Particle.ParticleType.NuE
    procs, _ = siren.utilities.load_processes(
        "CSMSDISSplines", primary_types=[nue],
        target_types=[siren.dataclasses.Particle.ParticleType.Nucleon],
        isoscalar=True, process_types=["NC" if args.nc else "CC"], shower_model_path=MODEL)

    inj = siren.injection.Injector()
    inj.number_of_events = args.n
    inj.detector_model = siren.utilities.load_detector("IceCube")
    inj.primary_type = nue
    inj.primary_interactions = procs[nue]
    inj.primary_injection_distributions = [
        siren.distributions.PrimaryMass(0),
        siren.distributions.PowerLaw(2, args.emin, args.emax),
        siren.distributions.IsotropicDirection(),
        siren.distributions.ColumnDepthPositionDistribution(
            600, 600.0, siren.distributions.LeptonDepthFunction()),
    ]
    print(f"injecting {args.n} {'NC' if args.nc else 'CC'} nu_e events ...")
    events, _ = GenerateEvents(inj)
    print(f"  got {len(events)} events\n")

    worst_had, worst_tot, n_sub = 0.0, 0.0, 0
    for ie, event in enumerate(events):
        rec = next(iter(event.tree)).record
        ip = dict(rec.interaction_parameters)
        subs = [np.array(m) for t, m in zip(rec.signature.secondary_types, rec.secondary_momenta)
                if int(t) == GAMMASUB and m[0] > 0.0]
        n_sub += len(subs)
        nhad = int(ip.get("had_count", 0))
        hsum = sum((four(ip, f"hadron{i}") for i in range(nhad)), np.zeros(4))
        had, lep, inv, mu = four(ip, "had"), four(ip, "lep"), four(ip, "invisible"), four(ip, "muon")
        res_had = np.abs(hsum + inv + mu - had).max()
        m_t = NUCLEON_MASS.get(int(ip.get("target_pdg", 0)), rec.target_mass)
        p_in = np.array(rec.primary_momentum, float) + np.array([m_t, 0.0, 0.0, 0.0])
        res_tot = np.abs(lep + had - p_in).max()
        worst_had, worst_tot = max(worst_had, res_had), max(worst_tot, res_tot)
        if ie < 5:
            print(f"ev {ie}: E_nu {rec.primary_momentum[0]:9.1f}  x {ip.get('bjorken_x', 0):.3g}  "
                  f"y {ip.get('bjorken_y', 0):.3f}  hadrons {nhad:2d}  subshowers {len(subs):2d}  "
                  f"invisible {inv[0]:.3g} GeV  muons {mu[0]:.3g} GeV  "
                  f"residuals {res_had:.1e} / {res_tot:.1e}")

    print(f"\nsubshowers produced                : {n_sub}")
    print(f"worst |Sum hadrons + invisible + muon - had| : {worst_had:.2e} GeV")
    print(f"worst |lep + had - (nu + target)|     : {worst_tot:.2e} GeV")
    if n_sub == 0:
        print("PROBLEM: no GammaSubshower secondaries")
    elif max(worst_had, worst_tot) > 1e-6 * args.emax:
        print("PROBLEM: 4-momentum not conserved")
    else:
        print("OK")


if __name__ == "__main__":
    main()
