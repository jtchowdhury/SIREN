#!/usr/bin/env python3
"""
pythia_dis_reference.py -- standalone Pythia8 DIS with the settings that
GammaDISFromSpline uses, as a reference for the validation plots.

  PythiaTruth   expected x, y, Q^2 distributions for a mixture of energies, with
                the same interface as gamma_dis_common.SplineTruth
  library       writes an event library (E_had, top-20 hadrons, momenta) in the
                format plot_gamma_dis_ccnc.py --pythia-lib reads, for CC or NC

The settings mirror GammaDISFromSpline::GetPythia: full DIS (W exchange for CC,
gamma/Z for NC), target at rest (proton or neutron, 10:8), MPI off, dipole recoil
in the initial-state shower, particles with c*tau < 10 mm decay except pi0 and tau,
the LHAPDF set given by --pdf-set.
The outgoing lepton is the most energetic final-state particle of its type; the
rest, without neutrinos, is the hadronic system.

    python pythia_dis_reference.py library --channel nc --out output/pythia_events_nc.npz
"""
import os
import argparse
import numpy as np

try:
    import pythia8
except ImportError:
    import pythia8mc as pythia8

PDF_SET = "LHAPDF6:HERAPDF20_NLO_EIG"
M_ISO = 0.9389187                       # isoscalar nucleon mass, as in SIREN
NEUTRINOS = (12, 14, 16)


def make_pythia(E_nu, beam, target, cc, pdf_set=PDF_SET, q2min=1.0, seed=1):
    p = pythia8.Pythia("", False)
    settings = [
        "WeakBosonExchange:ff2ff(t:W) = on" if cc else "WeakBosonExchange:ff2ff(t:gmZ) = on",
        "Beams:frameType = 2", f"Beams:idA = {beam}", f"Beams:idB = {target}",
        f"Beams:eA = {E_nu}", "Beams:eB = 0.",
        "PartonLevel:MPI = off", "PhaseSpace:mHatMin = 0.0", f"PhaseSpace:Q2Min = {q2min}",
        # keeps the scattered lepton's kinematics when the initial-state shower recoils
        "SpaceShower:dipoleRecoil = on",
        "ParticleDecays:limitTau0 = on", "ParticleDecays:tau0Max = 10.0",
        "111:mayDecay = off", "15:mayDecay = off",
        "Random:setSeed = on", f"Random:seed = {seed}",
        "Print:quiet = on", "Next:numberCount = 0",
    ]
    if pdf_set:
        settings += [f"PDF:pSet = {pdf_set}", "PDF:useHard = on", f"PDF:pHardSet = {pdf_set}"]
    for s in settings:
        if not p.readString(s):
            raise RuntimeError(f"Pythia rejected the setting '{s}'")
    if not p.init():
        raise RuntimeError(f"Pythia initialization failed (E_nu = {E_nu:g} GeV, target {target}). "
                           "If it is the PDF set, check LHAPDF_DATA_PATH or use --pdf-set ''.")
    return p


def events(E_nu, n, beam=12, cc=True, pdf_set=PDF_SET, q2min=1.0, seed=1):
    """Yield (lepton 4-momentum, [(pid, E, px, py, pz, E_kin)] of the hadrons,
    4-momentum of the rest of the hadronic system) for n events, protons and
    neutrons in the ratio 10:8. Same selection as GammaDISFromSpline: the lepton is
    the hard-process one followed to its final copy, decay neutrinos and muons are
    not in the hadron list but are in the rest."""
    lep_id = beam - 1 if (cc and beam > 0) else (beam + 1 if cc else beam)
    n_p = int(round(n * 10 / 18))
    for target, nt in ((2212, n_p), (2112, n - n_p)):
        if nt <= 0:
            continue
        py = make_pythia(E_nu, beam, target, cc, pdf_set, q2min, seed + target)
        done = 0
        while done < nt:
            if not py.next():
                continue
            ev = py.event
            i_lep = -1
            for i in range(ev.size()):
                if ev[i].statusAbs() == 23 and ev[i].id() == lep_id:
                    j = ev[i].iBotCopyId()
                    if ev[j].isFinal():
                        i_lep = j
                    break
            if i_lep < 0:
                continue
            had, rest = [], np.zeros(4)
            for i in range(ev.size()):
                if not ev[i].isFinal() or i == i_lep:
                    continue
                p = ev[i]
                if abs(p.id()) in NEUTRINOS or abs(p.id()) == 13:
                    rest += [p.e(), p.px(), p.py(), p.pz()]
                    continue
                had.append((p.id(), p.e(), p.px(), p.py(), p.pz(), max(p.e() - p.m(), 0.0)))
            l = ev[i_lep]
            yield np.array([l.e(), l.px(), l.py(), l.pz()]), had, rest
            done += 1


def lepton_kinematics(E_nu, lep):
    """Q^2 and y from the lepton, neutrino along +z (as in GammaDISFromSpline)."""
    q = np.array([E_nu, 0.0, 0.0, E_nu]) - lep
    Q2 = -(q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2)
    return Q2, 1.0 - lep[0] / E_nu


class PythiaTruth:
    """x, y, Q^2 of standalone Pythia DIS, cached per energy."""

    label = "Pythia DIS (standalone)"

    def __init__(self, cache=None, n=20000, beam=12, cc=True, pdf_set=PDF_SET, q2min=1.0, seed=1):
        self.n, self.beam, self.cc, self.pdf_set, self.q2min, self.seed = n, beam, cc, pdf_set, q2min, seed
        self.tag = f"{n}|{beam}|{int(cc)}|{pdf_set}|{q2min}|{seed}"
        self.cache_path = cache
        self._ev = {}
        if cache and os.path.exists(cache):
            with np.load(cache) as d:
                if "tag" in d.files and str(d["tag"]) == self.tag:
                    for k in d.files:
                        if k.endswith("|Q2"):
                            E = k[:-3]
                            self._ev[float(E)] = (d[k], d[E + "|y"])

    def _events(self, E):
        key = float(f"{float(E):.6e}")
        if key not in self._ev:
            print(f"  Pythia reference: {self.n} events at E_nu = {key:g} GeV")
            Q2, y = [], []
            for lep, _, _ in events(key, self.n, self.beam, self.cc, self.pdf_set, self.q2min, self.seed):
                q2, yy = lepton_kinematics(key, lep)
                Q2.append(q2); y.append(yy)
            self._ev[key] = (np.array(Q2), np.array(y))
            self._save()
        return self._ev[key]

    def _save(self):
        if not self.cache_path:
            return
        arrs = {}
        for E, (Q2, y) in self._ev.items():
            arrs[f"{E:.6e}|Q2"] = Q2
            arrs[f"{E:.6e}|y"] = y
        np.savez(self.cache_path, tag=np.array(self.tag), **arrs)

    def _vars(self, E, M):
        Q2, y = self._events(E)
        ok = y > 0
        Q2, y = Q2[ok], y[ok]
        return {"x": Q2 / (2.0 * M * E * y), "y": y, "Q2": Q2}

    def expected_1d(self, var, bins, comps, M):
        tot, wsum = np.zeros(len(bins) - 1), 0.0
        for E, w in comps:
            v = self._vars(E, M)[var]
            tot += w * np.histogram(v, bins)[0] / len(v)
            wsum += w
        return tot / wsum

    def expected_2d(self, va, vb, ba, bb, comps, M):
        tot, wsum = np.zeros((len(ba) - 1, len(bb) - 1)), 0.0
        for E, w in comps:
            v = self._vars(E, M)
            tot += w * np.histogram2d(v[va], v[vb], [ba, bb])[0] / len(v[va])
            wsum += w
        return tot / wsum

    def mean(self, var, E, M):
        return float(np.mean(self._vars(E, M)[var]))


def write_library(path, cc, beam, energies, n, pdf_set, q2min, seed):
    # hadrons sorted by kinetic energy, as in GammaDISFromSpline; rest_ekin is the
    # kinetic energy of the hadrons beyond the stored 20
    rec = {k: [] for k in ("E_nu", "E_had", "top20_pids", "top20_energies", "top20_ekin",
                           "top20_momenta", "rest_ekin", "p_had", "p_lepton")}
    for E in energies:
        print(f"  {n} {'CC' if cc else 'NC'} events at E_nu = {E:g} GeV")
        for lep, had, rest in events(E, n, beam, cc, pdf_set, q2min, seed):
            had.sort(key=lambda h: -h[5])
            top = had[:20]
            pad = 20 - len(top)
            p_had = (np.sum([h[1:5] for h in had], axis=0) if had else np.zeros(4)) + rest
            rec["E_nu"].append(E)
            rec["E_had"].append(p_had[0])
            rec["top20_pids"].append([h[0] for h in top] + [0] * pad)
            rec["top20_energies"].append([h[1] for h in top] + [0.0] * pad)
            rec["top20_ekin"].append([h[5] for h in top] + [0.0] * pad)
            rec["top20_momenta"].append([list(h[1:5]) for h in top] + [[0.0] * 4] * pad)
            rec["rest_ekin"].append(sum(h[5] for h in had[20:]))
            rec["p_had"].append(p_had)
            rec["p_lepton"].append(lep)
    out = {k: np.array(v, dtype=np.int64 if k == "top20_pids" else float) for k, v in rec.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path}: {len(out['E_had'])} events")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lib = sub.add_parser("library", help="write an event library for plot_gamma_dis_ccnc.py")
    lib.add_argument("--channel", choices=["cc", "nc"], required=True)
    lib.add_argument("--out", required=True)
    lib.add_argument("--n", type=int, default=5000, help="events per energy")
    lib.add_argument("--energies", type=float, nargs="+", default=list(np.logspace(2, 7, 11)))
    lib.add_argument("--beam", type=int, default=12)
    lib.add_argument("--pdf-set", default=PDF_SET, help="'' for Pythia's built-in PDF")
    lib.add_argument("--q2min", type=float, default=1.0)
    lib.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "library":
        write_library(args.out, args.channel == "cc", args.beam, args.energies, args.n,
                      args.pdf_set, args.q2min, args.seed)


if __name__ == "__main__":
    main()
