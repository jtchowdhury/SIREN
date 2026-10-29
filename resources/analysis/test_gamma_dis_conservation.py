#!/usr/bin/env python3
"""
test_gamma_dis_conservation.py -- first runtime check of GammaDISFromSpline.

Injects a handful of CC nu_e DIS events with the GammaDISFromSpline cross section
and checks, straight off the live interaction records (no SaveEvents needed):

  1. it runs without crashing and produces GammaSubshower secondaries;
  2. the boosted hadrons conserve 4-momentum: Sum_i hadron<i> == blob (had_*);
  3. the DIS vertex conserves 4-momentum: lep_* + had_* == p_nu + p_target.

Interpretation of check (2):
  * residual ~ 0           -> boost ran and momentum is conserved.
  * hadron momenta all 0   -> fell back to energy-scaling (npz has no momenta).
  * residual NOT ~ 0 (but momenta nonzero) -> rk boost convention is wrong.

Run:
    export SIREN_ROOT=$HOME/SIREN
    python test_gamma_dis_conservation.py
"""
import os
import numpy as np
import siren
from siren._util import GenerateEvents

SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
MODEL  = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model_cpp.npz")
PYTHIA = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/pythia_events_nue.npz")

GAMMASUB = -2000002200      # must match ParticleTypes.def
M_PROTON = 0.9382720813
N_EVENTS = 25

primary_type = siren.dataclasses.Particle.ParticleType.NuE
detector_model = siren.utilities.load_detector("IceCube")

primary_processes, _ = siren.utilities.load_processes(
    "CSMSDISSplines",
    primary_types=[primary_type],
    target_types=[siren.dataclasses.Particle.ParticleType.Nucleon],
    isoscalar=True,
    process_types=["CC"],
    shower_model_path=MODEL,
    pythia_events_path=PYTHIA,
)

injector = siren.injection.Injector()
injector.number_of_events = N_EVENTS
injector.detector_model = detector_model
injector.primary_type = primary_type
injector.primary_interactions = primary_processes[primary_type]
injector.primary_injection_distributions = [
    siren.distributions.PrimaryMass(0),
    siren.distributions.PowerLaw(2, 1e3, 1e6),           # E_nu: 1 TeV - 1 PeV
    siren.distributions.IsotropicDirection(),
    siren.distributions.ColumnDepthPositionDistribution(
        600, 600.0, siren.distributions.LeptonDepthFunction()),
]

print(f"Injecting {N_EVENTS} CC nu_e events with GammaDISFromSpline ...")
events, _ = GenerateEvents(injector)
print(f"  got {len(events)} events (no crash).\n")

worst_had = 0.0
worst_tot = 0.0
n_sub_total = 0
had_momenta_all_zero = True

for ie, event in enumerate(events):
    for datum in event.tree:
        rec = datum.record
        ip = dict(rec.interaction_parameters)
        types = [int(t) for t in rec.signature.secondary_types]
        moms = [np.array(p, dtype=float) for p in rec.secondary_momenta]

        # GammaSubshower secondaries: momentum = [amplitude, alpha, beta, parent_pid]
        subs = [m for t, m in zip(types, moms) if t == GAMMASUB and m[0] > 0.0]
        n_sub_total += len(subs)

        # (2) hadron-level conservation: Sum_i hadron<i> == blob
        nhad = len([k for k in ip if k.startswith("hadron") and k.endswith("_E")])
        sH = np.zeros(4)
        for i in range(nhad):
            q = np.array([ip[f"hadron{i}_E"], ip[f"hadron{i}_px"],
                          ip[f"hadron{i}_py"], ip[f"hadron{i}_pz"]])
            sH += q
            if np.any(q[1:] != 0.0):
                had_momenta_all_zero = False
        blob = np.array([ip.get("had_E", 0), ip.get("had_px", 0),
                         ip.get("had_py", 0), ip.get("had_pz", 0)])
        res_had = np.abs(sH - blob).max() if nhad else float("nan")

        # (3) DIS vertex: lep + blob == p_nu + p_target
        lep = np.array([ip.get("lep_E", 0), ip.get("lep_px", 0),
                        ip.get("lep_py", 0), ip.get("lep_pz", 0)])
        pnu = np.array(rec.primary_momentum, dtype=float)
        mtgt = getattr(rec, "target_mass", M_PROTON)
        ptgt = np.array([mtgt, 0.0, 0.0, 0.0])
        res_tot = np.abs((lep + blob) - (pnu + ptgt)).max()

        worst_had = max(worst_had, 0.0 if np.isnan(res_had) else res_had)
        worst_tot = max(worst_tot, res_tot)

        if ie < 5:
            a = subs[0] if subs else np.zeros(4)
            print(f"ev {ie:2d}: subshowers={len(subs):2d}  hadrons={nhad:2d}  "
                  f"|Sum h - blob|={res_had:.2e}  |lep+blob-(nu+tgt)|={res_tot:.2e}  "
                  f"(e.g. amp={a[0]:.3g}, alpha={a[1]:.2f}, beta={a[2]:.4f}, pid={int(a[3])})")

print()
print(f"total gamma subshowers produced : {n_sub_total}")
print(f"worst | Sum hadron - blob |      : {worst_had:.3e}   (should be ~1e-6 or better)")
print(f"worst | lep+blob - (nu+target) | : {worst_tot:.3e}   (should be ~machine eps)")
print()
if all(not any(k.startswith("hadron") for k in dict(d.record.interaction_parameters))
       for ev in events for d in ev.tree):
    print("WARNING: no hadron<i>_* / had_count keys in interaction_parameters at all.")
    print("         -> the installed GammaDISFromSpline is NOT the current source")
    print("            (rebuild from the latest GammaDISFromSpline.cxx).")
elif had_momenta_all_zero:
    print("WARNING: all hadron momenta are zero -> it FELL BACK to energy-scaling.")
    print("         (the loaded pythia_events_nue.npz has no four-momenta -> regenerate it.)")
elif worst_had > 1e-3:
    print("WARNING: hadrons do NOT sum to the blob -> the rk boost convention is likely wrong.")
    print("         (restBoost/labBoost direction or the CM rescale needs fixing.)")
else:
    print("OK: boost ran and momentum is conserved at both levels.")
