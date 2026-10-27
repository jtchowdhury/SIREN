#!/usr/bin/env python3
"""
test_gamma_subshowers.py  --  end-to-end check of the gamma_subshowers output.

Runs a small CC nu_e DIS injection in SIREN (100 events), which produces the
lepton + Hadrons blob per event; SaveEvents then develops each blob into gamma
subshowers and writes them to test.hdf5. Validate with:

    python test_gamma_subshowers.py
    python check_subshowers.py test.hdf5

REQUIRES SIREN to be importable (this is a real injection). It's tiny (100
events, a few seconds) and needs no SLURM. If you can't import siren locally,
run test_gamma_subshowers_local.py instead -- it exercises the same
gamma_subshowers path with mock events and no siren.
"""
import os
import siren
from siren._util import GenerateEvents, SaveEvents

# where your model + pythia store live (edit if different)
SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
# the C++ GammaDISFromSpline reads the NUMERIC-ONLY repack (convert_model_for_cpp.py),
# not the original shower_model.npz (whose __meta__ is a JSON/unicode array cnpy can't read)
MODEL_NPZ  = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model_cpp.npz")
PYTHIA_NPZ = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/pythia_events_nue.npz")

N_EVENTS = 100
primary_type = siren.dataclasses.Particle.ParticleType.NuE

detector_model = siren.utilities.load_detector("IceCube")

primary_processes, _ = siren.utilities.load_processes(
    "CSMSDISSplines",
    primary_types=[primary_type],
    target_types=[siren.dataclasses.Particle.ParticleType.Nucleon],
    isoscalar=True,
    process_types=["CC"],
    shower_model_path=MODEL_NPZ,      # -> builds GammaDISFromSpline
    pythia_events_path=PYTHIA_NPZ,
)
primary_cross_sections = primary_processes[primary_type]

injector = siren.injection.Injector()
injector.number_of_events = N_EVENTS
injector.detector_model = detector_model
injector.primary_type = primary_type
injector.primary_interactions = primary_cross_sections
injector.primary_injection_distributions = [
    siren.distributions.PrimaryMass(0),
    siren.distributions.PowerLaw(2, 1e3, 1e6),          # E_nu: 1 TeV - 1 PeV
    siren.distributions.IsotropicDirection(),
    siren.distributions.ColumnDepthPositionDistribution(
        600, 600.0, siren.distributions.LeptonDepthFunction()),
]

print(f"Injecting {N_EVENTS} CC nu_e DIS events ...")
events, gen_times = GenerateEvents(injector)

print("Developing blobs + saving (test.hdf5) ...")
# NOTE: with GammaDISFromSpline, the subshowers are developed in C++ during the
# injection and stored on each record's interaction_parameters. SaveEvents just
# reads them back into the gamma_subshowers HDF5 field -- no model paths needed here.
SaveEvents(
    events,
    weighter=None,
    gen_times=gen_times,
    output_filename="test",
)
print("\nDone. Now run:  python check_subshowers.py test.hdf5")
