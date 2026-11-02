#!/usr/bin/env python3
"""
gamma_dis_samples.py -- generate the event samples for the GammaDISFromSpline
validation plots and save one compact .npz per sample in --outdir.

All samples: CC nu_e on isoscalar nucleons, IceCube, isotropic directions.

  mono_E<E>_gamma.npz / mono_E<E>_dis.npz   one fixed E_nu, GammaDIS / DISFromSpline
  range_gamma.npz     / range_dis.npz       generated ~E^-2, weighted to E^-2
  reweight_gamma.npz                        generated ~E^-1, weighted to E^-2
                                            (the "test" for the reweighting check;
                                             range_gamma.npz is its "reference")

Stored per event: E_nu, p_nu, target_mass, bjorken x, y, weight (NaN if none).
GammaDIS samples additionally store the lepton and blob 4-momenta, every boosted
hadron (pid + 4-momentum), every real subshower (amplitude, alpha, beta,
parent index), and the cross section from the GammaDIS override and from plain
DISFromSpline at the same kinematics (reweighting check E2).

Usage (compute node; ~1e4 events per sample is a short batch job):
    export SIREN_ROOT=$HOME/SIREN
    python gamma_dis_samples.py --which all --n 10000
    python gamma_dis_samples.py --which mono --energies 1e3 1e4 1e5 1e6 --n 10000
"""
import os
import argparse
import numpy as np
import siren
from siren._util import GenerateEvents

SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
SHOWERS = os.path.join(SIREN_ROOT, "resources/showers/GammaShowerModel-v1.0")
MODEL = os.path.join(SHOWERS, "shower_model_cpp.npz")
PYTHIA = os.path.join(SHOWERS, "pythia_events_nue.npz")

PT = siren.dataclasses.Particle.ParticleType
NUE, NUCLEON = PT.NuE, PT.Nucleon
GAMMASUB = -2000002200
ME = siren.utilities.Constants.electronMass


def load_xs(gamma):
    kw = dict(primary_types=[NUE], target_types=[NUCLEON], isoscalar=True,
              process_types=["CC"])
    if gamma:
        kw.update(shower_model_path=MODEL, pythia_events_path=PYTHIA)
    procs, _ = siren.utilities.load_processes("CSMSDISSplines", **kw)
    return procs[NUE]


def make_injector(xs, edist, n, detector, seed):
    inj = siren.injection.Injector(seed=seed)
    inj.number_of_events = n
    inj.detector_model = detector
    inj.primary_type = NUE
    inj.primary_interactions = xs
    inj.primary_injection_distributions = [
        siren.distributions.PrimaryMass(0),
        edist,
        siren.distributions.IsotropicDirection(),
        siren.distributions.ColumnDepthPositionDistribution(
            600, 600.0, siren.distributions.LeptonDepthFunction()),
    ]
    return inj


def make_weighter(inj, xs, detector, emin, emax):
    w = siren.injection.Weighter()
    w.injectors = [inj]
    w.detector_model = detector
    w.primary_type = NUE
    w.primary_interactions = xs
    w.primary_physical_distributions = [
        siren.distributions.PowerLaw(2, emin, emax),
        siren.distributions.IsotropicDirection(),
    ]
    return w


def four(ip, prefix):
    return [ip[prefix + "_E"], ip[prefix + "_px"], ip[prefix + "_py"], ip[prefix + "_pz"]]


def extract(events, gamma, weighter=None, gamma_xs=None, dis_ref=None):
    c = {k: [] for k in ("E_nu", "p_nu", "target_mass", "x", "y", "weight")}
    if gamma:
        for k in ("lep", "blob", "had_n", "had_pid", "had_p", "sub_n", "sub_amp",
                  "sub_alpha", "sub_beta", "sub_parent", "xs_override", "xs_base"):
            c[k] = []
    for ev in events:
        rec = next(iter(ev.tree)).record          # the primary interaction
        ip = dict(rec.interaction_parameters)
        pnu = np.array(rec.primary_momentum, float)
        c["E_nu"].append(pnu[0])
        c["p_nu"].append(pnu)
        c["target_mass"].append(rec.target_mass)
        c["x"].append(ip.get("bjorken_x", np.nan))
        c["y"].append(ip.get("bjorken_y", np.nan))
        c["weight"].append(weighter(ev) if weighter is not None else np.nan)
        if not gamma:
            continue

        lep = np.array(four(ip, "lep"))
        c["lep"].append(lep)
        c["blob"].append(four(ip, "had"))
        n = 0
        while f"hadron{n}_E" in ip:
            c["had_pid"].append(int(ip[f"hadron{n}_pid"]))
            c["had_p"].append(four(ip, f"hadron{n}"))
            n += 1
        c["had_n"].append(n)

        k = 0
        for t, m in zip(rec.signature.secondary_types, rec.secondary_momenta):
            if int(t) == GAMMASUB and m[0] > 0.0:      # skip padding (amplitude 0)
                c["sub_amp"].append(m[0]); c["sub_alpha"].append(m[1])
                c["sub_beta"].append(m[2]); c["sub_parent"].append(int(round(m[3])))
                k += 1
        c["sub_n"].append(k)

        # E2: GammaDIS override vs plain DISFromSpline at identical kinematics
        # (base formulas: q = p_nu - p_lep, Q2 = -q.q, y = 1 - E_lep/E_nu, x = Q2/(2 M q0))
        M = rec.target_mass
        q = pnu - lep
        Q2 = -(q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2)
        yb = 1.0 - lep[0] / pnu[0]
        xb = Q2 / (2.0 * M * q[0])
        c["xs_override"].append(gamma_xs.DifferentialCrossSection(rec))
        c["xs_base"].append(dis_ref.DifferentialCrossSection(pnu[0], xb, yb, ME, Q2))

    out = {}
    for key, v in c.items():
        if key in ("had_n", "sub_n"):
            out[key.replace("_n", "_off")] = np.concatenate([[0], np.cumsum(v)]).astype(np.int64)
        elif key in ("had_pid", "sub_parent"):
            out[key] = np.array(v, dtype=np.int64)
        else:
            out[key] = np.array(v, dtype=float)
    return out


def run(name, gamma, edist, n, seed, outdir, detector, weight_range=None,
        dis_ref=None):
    xs = load_xs(gamma)
    inj = make_injector(xs, edist, n, detector, seed)
    print(f"\n== {name}: {n} events ({'GammaDIS' if gamma else 'DISFromSpline'}) ==")
    events, _ = GenerateEvents(inj)
    weighter = make_weighter(inj, xs, detector, *weight_range) if weight_range else None
    d = extract(events, gamma, weighter, xs[0] if gamma else None, dis_ref)
    path = os.path.join(outdir, name + ".npz")
    np.savez(path, **d)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["mono", "range", "reweight", "all"], default="all")
    ap.add_argument("--n", type=int, default=10000, help="events per sample")
    ap.add_argument("--energies", type=float, nargs="+", default=[1e3, 1e4, 1e5, 1e6],
                    help="E_nu [GeV] for the monochromatic samples")
    ap.add_argument("--emin", type=float, default=1e3)
    ap.add_argument("--emax", type=float, default=1e6)
    ap.add_argument("--outdir", default="output/gamma_dis_samples")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    detector = siren.utilities.load_detector("IceCube")
    dis_ref = load_xs(False)[0]       # plain DISFromSpline, for the E2 comparison
    wr = (args.emin, args.emax)
    seed = 1000

    if args.which in ("mono", "all"):
        for E in args.energies:
            for gamma in (True, False):
                seed += 1
                run(f"mono_E{E:.0e}_{'gamma' if gamma else 'dis'}", gamma,
                    siren.distributions.Monoenergetic(E), args.n, seed,
                    args.outdir, detector, dis_ref=dis_ref)
    if args.which in ("range", "all"):
        for gamma in (True, False):
            seed += 1
            run(f"range_{'gamma' if gamma else 'dis'}", gamma,
                siren.distributions.PowerLaw(2, args.emin, args.emax), args.n, seed,
                args.outdir, detector, weight_range=wr, dis_ref=dis_ref)
    if args.which in ("reweight", "all"):
        seed += 1
        run("reweight_gamma", True,
            siren.distributions.PowerLaw(1, args.emin, args.emax), args.n, seed,
            args.outdir, detector, weight_range=wr, dis_ref=dis_ref)


if __name__ == "__main__":
    main()
