#!/usr/bin/env python3
"""
test_gamma_subshowers_local.py  --  validate the gamma_subshowers path WITHOUT siren.

Fabricates 100 mock CC nu_e interactions (draws E_nu ~ power law, y ~ uniform, so
E_had = y*E_nu and the electron carries (1-y)*E_nu), develops each blob into
gamma subshowers with BlobDeveloper, builds the SAME awkward 'gamma_subshowers'
record SaveEvents builds, and writes it to test_local.hdf5 in the same HDF5
layout SaveEvents uses (Events group + form/length attrs). Then:

    python test_gamma_subshowers_local.py
    python check_subshowers.py test_local.hdf5

Needs only the siren-dev env (numpy, scipy, awkward, h5py) -- NOT siren itself.
This checks the model/store loading, the blob development, the awkward nesting,
and the checker; it does not exercise the real DISFromSpline kinematics.
"""
import os
import sys
import numpy as np
import h5py
import awkward as ak

# import hadronic_showers from the repo (no siren install needed)
SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
sys.path.insert(0, os.path.join(SIREN_ROOT, "python"))
import hadronic_showers as hs

MODEL_NPZ  = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/shower_model.npz")
PYTHIA_NPZ = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0/pythia_events_nue.npz")

N_EVENTS = 100
rng = np.random.default_rng(0)

dev = hs.BlobDeveloper(model=MODEL_NPZ, pythia_source=PYTHIA_NPZ)

# one interaction per event; mimic CC nu_e kinematics
datasets = {"event_weight": [], "num_interactions": [], "gamma_subshowers": []}
for _ in range(N_EVENTS):
    E_nu = 10 ** rng.uniform(3, 6)             # 1 TeV - 1 PeV
    y = rng.uniform(0.05, 0.95)
    E_had = y * E_nu
    E_e = (1 - y) * E_nu                        # scattered electron (CC nu_e)
    gs = dev.develop(E_had, rng=rng, lepton=("em", E_e))
    datasets["event_weight"].append(0.0)
    datasets["num_interactions"].append(1)
    datasets["gamma_subshowers"].append([gs])   # list over interactions (one here)

# build + write exactly like SaveEvents
ak_array = ak.Array(datasets)
with h5py.File("test_local.hdf5", "w") as fout:
    group = fout.create_group("Events")
    form, length, _ = ak.to_buffers(ak.to_packed(ak_array), container=group)
    group.attrs["form"] = form.to_json()
    group.attrs["length"] = length

n_sub = sum(len(gs["subshower_parent"]) for ev in datasets["gamma_subshowers"] for gs in ev)
print(f"wrote test_local.hdf5: {N_EVENTS} events, {n_sub} gamma subshowers")
print("Now run:  python check_subshowers.py test_local.hdf5")
