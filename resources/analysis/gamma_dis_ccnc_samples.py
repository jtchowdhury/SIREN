#!/usr/bin/env python3
"""
gamma_dis_ccnc_samples.py -- GammaDISFromSpline samples for the CC vs NC
event-shower plots (plot_gamma_dis_ccnc.py).

nu_e on isoscalar nucleons, IceCube, isotropic. Both channels use the SAME (CC)
PYTHIA library for the hadronic breakdown; NC only differs in that the outgoing
lepton is the neutrino (kept as a real, unshowered track), so there is no
electron subshower.

  ccnc_mono_E<E>_{cc,nc}.npz   one fixed E_nu each   (spaghetti plot at fixed E_nu)
  ccnc_range_{cc,nc}.npz       E_nu ~ E^-1 over [emin, emax]
                               (spaghetti plot at fixed E_had by selection, and
                                L2/KS vs E_had)

Stored per event: E_nu, x, y, lepton 4-mom (lep), blob 4-mom (blob = had_*),
every boosted hadron (had_pid, had_p) and every real subshower (sub_amp,
sub_alpha, sub_beta, sub_parent; parent -1 = the electron, CC only).

Usage (compute node):
    export SIREN_ROOT=$HOME/SIREN
    python gamma_dis_ccnc_samples.py --n-mono 300 --n-range 20000
"""
import os
import argparse
import numpy as np
import siren
from siren._util import GenerateEvents

from gamma_dis_samples import MODEL, PYTHIA, NUE, NUCLEON, GAMMASUB, make_injector, four

ITYPE = {"cc": 1, "nc": 2}   # DISFromSpline interaction_type_


def load_xs(channel):
    procs, _ = siren.utilities.load_processes(
        "CSMSDISSplines", primary_types=[NUE], target_types=[NUCLEON], isoscalar=True,
        process_types=[channel.upper()], shower_model_path=MODEL, pythia_events_path=PYTHIA)
    xs = procs[NUE]
    it = getattr(xs[0], "GetInteractionType", None)
    if it is not None and it() != ITYPE[channel]:
        raise RuntimeError(f"{channel}: spline says interaction type {it()}, "
                           f"expected {ITYPE[channel]}")
    return xs


def extract(events, channel):
    c = {k: [] for k in ("E_nu", "x", "y", "lep", "blob", "had_n", "had_pid", "had_p",
                         "sub_n", "sub_amp", "sub_alpha", "sub_beta", "sub_parent")}
    for i, ev in enumerate(events):
        rec = next(iter(ev.tree)).record
        ip = dict(rec.interaction_parameters)
        types = [int(t) for t in rec.signature.secondary_types]
        if i == 0:   # one-line sanity check of the signature
            real = [t for t in types if t != GAMMASUB]
            print(f"  {channel}: real (non-subshower) secondaries = {real}  "
                  f"(expect [] for CC nu_e, [{int(NUE)}] for NC)")
        c["E_nu"].append(rec.primary_momentum[0])
        c["x"].append(ip.get("bjorken_x", np.nan))
        c["y"].append(ip.get("bjorken_y", np.nan))
        c["lep"].append(four(ip, "lep"))
        c["blob"].append(four(ip, "had"))
        n = 0
        while f"hadron{n}_E" in ip:
            c["had_pid"].append(int(ip[f"hadron{n}_pid"]))
            c["had_p"].append(four(ip, f"hadron{n}"))
            n += 1
        c["had_n"].append(n)
        k = 0
        for t, m in zip(types, rec.secondary_momenta):
            if t == GAMMASUB and m[0] > 0.0:          # skip padding (amplitude 0)
                c["sub_amp"].append(m[0]); c["sub_alpha"].append(m[1])
                c["sub_beta"].append(m[2]); c["sub_parent"].append(int(round(m[3])))
                k += 1
        c["sub_n"].append(k)
        if channel == "nc" and k and min(c["sub_parent"][-k:]) < 0:
            raise RuntimeError("NC event has an electron subshower (parent -1): "
                               "the C++ NC handling is not active -- rebuild SIREN")
    out = {"channel": np.array(channel)}
    for key, v in c.items():
        if key in ("had_n", "sub_n"):
            out[key.replace("_n", "_off")] = np.concatenate([[0], np.cumsum(v)]).astype(np.int64)
        elif key in ("had_pid", "sub_parent"):
            out[key] = np.array(v, dtype=np.int64)
        else:
            out[key] = np.array(v, dtype=float)
    return out


def run(name, channel, edist, n, seed, outdir, detector):
    xs = load_xs(channel)
    inj = make_injector(xs, edist, n, detector, seed)
    print(f"\n== {name}: {n} events ({channel.upper()}) ==")
    events, _ = GenerateEvents(inj)
    path = os.path.join(outdir, name + ".npz")
    np.savez(path, **extract(events, channel))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["mono", "range", "all"], default="all")
    ap.add_argument("--energies", type=float, nargs="+", default=[1e3, 1e4, 1e5],
                    help="E_nu [GeV] for the fixed-E_nu samples")
    ap.add_argument("--n-mono", type=int, default=300, help="events per fixed-E_nu sample")
    ap.add_argument("--n-range", type=int, default=20000,
                    help="events per range sample (fixed-E_had plots select a window from it)")
    ap.add_argument("--emin", type=float, default=1e2)
    ap.add_argument("--emax", type=float, default=1e6)
    ap.add_argument("--outdir", default="output/gamma_dis_samples")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    detector = siren.utilities.load_detector("IceCube")
    seed = 5000
    for ch in ("cc", "nc"):
        if args.which in ("mono", "all"):
            for E in args.energies:
                seed += 1
                run(f"ccnc_mono_E{E:.0e}_{ch}", ch, siren.distributions.Monoenergetic(E),
                    args.n_mono, seed, args.outdir, detector)
        if args.which in ("range", "all"):
            seed += 1
            run(f"ccnc_range_{ch}", ch,
                siren.distributions.PowerLaw(1, args.emin, args.emax),
                args.n_range, seed, args.outdir, detector)


if __name__ == "__main__":
    main()
