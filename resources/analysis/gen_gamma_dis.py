#!/usr/bin/env python3
"""
gen_gamma_dis.py -- generate N GammaDISFromSpline events and write:
    <out>.hdf5, <out>.parquet   (via SaveEvents, with all the new fields)
    <out>_secondaries.csv       (one row per secondary; GammaSubshower momenta
                                  are [amplitude, alpha, beta, parent_pid])
    <out>_params.csv            (one row per interaction_parameters entry:
                                  energy/bjorken_x/y, had_*, lep_*, hadron<i>_*)

Usage:
    export SIREN_ROOT=$HOME/SIREN
    python gen_gamma_dis.py            # 10 events
    python gen_gamma_dis.py 100        # N events
"""
import os
import sys
import csv
import numpy as np
import siren
from siren._util import GenerateEvents, SaveEvents

SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
MODEL  = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model_cpp.npz")
PYTHIA = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/pythia_events_nue.npz")
N   = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = "gammadis_test"
GAMMASUB = -2000002200

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
injector.number_of_events = N
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

print(f"Injecting {N} CC nu_e events with GammaDISFromSpline ...")
events, gen_times = GenerateEvents(injector)

print("Writing HDF5 + parquet (SaveEvents) ...")
SaveEvents(events, weighter=None, gen_times=gen_times, output_filename=OUT)

# --- flat CSVs straight from the live records (no awkward round-trip) ---------
with open(OUT + "_secondaries.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event", "interaction", "sec_idx", "type", "is_gammasub",
                "mom0_or_amp", "mom1_or_alpha", "mom2_or_beta", "mom3_or_pid"])
    for ie, event in enumerate(events):
        for ii, datum in enumerate(event.tree):
            types = list(datum.record.signature.secondary_types)
            moms  = list(datum.record.secondary_momenta)
            for si, (t, m) in enumerate(zip(types, moms)):
                t = int(t); m = [float(x) for x in m]
                w.writerow([ie, ii, si, t, int(t == GAMMASUB),
                            m[0], m[1], m[2], m[3]])

with open(OUT + "_params.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event", "interaction", "key", "value"])
    for ie, event in enumerate(events):
        for ii, datum in enumerate(event.tree):
            ip = dict(datum.record.interaction_parameters)
            for k in sorted(ip.keys()):
                w.writerow([ie, ii, str(k), float(ip[k])])

print(f"wrote:\n  {OUT}.hdf5\n  {OUT}.parquet\n  {OUT}_secondaries.csv\n  {OUT}_params.csv")
