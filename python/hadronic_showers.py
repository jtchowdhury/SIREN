"""
hadronic_showers.py  --  fast generative hadronic shower sampler for SIREN
==========================================================================
Sample the longitudinal Cherenkov profile of a hadronic shower in ice from a
model calibrated to Geant4, with NO Geant4 at run time. Each shower's profile
is a sum of gamma kernels whose parameters are drawn from energy-interpolated
distributions, so event-to-event fluctuations are preserved.

The model data is a portable .npz produced by
geant4_shower/python_scripts/convert_model_to_npz.py (plain numbers, no pickle).

Typical use
-----------
    from siren import hadronic_showers as hs      # or: import hadronic_showers as hs

    model = hs.load_model("shower_model.npz")

    s  = model.sample("pip", 1000.0)              # one Shower
    ss = model.sample_many("pi0", 1000.0, n=100)  # EM shower (pi0) x100
    ev = model.sample_event("example_final_state.csv")   # one DIS event + composite

    hs.plot_many(model, "pip", 1000.0, n=100)
    hs.plot_event(model, "example_final_state.csv", composite=True)

A Shower is a small object (species, energy, depth grid z [cm], photons/bin,
and the sampled mixture parameters). An Event bundles the per-hadron Showers
with their summed composite profile.

Species may be given by name ("pip", "pi0", "Kp", ...) or by PDG code
(211, 111, 321, ...), so SIREN ParticleType codes drop in directly later.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------------------
#  species bookkeeping (must match geant4_shower/.../shower_gamma_model.SPECIES)
# --------------------------------------------------------------------------
SPECIES = [
    (111, "pi0"), (211, "pip"), (-211, "pim"),
    (321, "Kp"), (-321, "Km"), (310, "KS"), (130, "KL"),
    (2212, "p"), (2112, "n"),
]
NAME_TO_PID = {name: pid for pid, name in SPECIES}
PID_TO_NAME = {pid: name for pid, name in SPECIES}
SPECIES_LATEX = {
    "pip": r"$\pi^{+}$", "pim": r"$\pi^{-}$", "pi0": r"$\pi^{0}$",
    "Kp": r"$K^{+}$", "Km": r"$K^{-}$", "KS": r"$K^{0}_{S}$", "KL": r"$K^{0}_{L}$",
    "p": r"$p$", "n": r"$n$",
}

# default bundled model location, if you place it under resources/ as suggested
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    _HERE, "..", "resources", "showers",
    "GammaShowerModel-v1.0", "shower_model.npz",
)

# numpy>=2 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

_CLIP_SIGMA = 2.5      # truncate the sampled Gaussian at +-2.5 sigma (matches training)
_COV_RIDGE = 1e-3      # PSD floor on covariance eigenvalues (matches training)


# --------------------------------------------------------------------------
#  sampling math -- ported verbatim from shower_gamma_model so draws are identical
# --------------------------------------------------------------------------
def _kernel(x, alpha, beta):
    """Unit-area gamma kernel  beta^a x^(a-1) e^(-b x) / Gamma(a)  (log-space)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    pos = x > 0
    lk = (alpha * np.log(beta) + (alpha - 1.0) * np.log(x[pos])
          - beta * x[pos] - gammaln(alpha))
    out[pos] = np.exp(lk)
    return out


def _comp_from_z(lc, ls):
    """(log centroid, log width) -> (alpha, beta). Inverse of the training encoding."""
    c = np.exp(np.asarray(lc, float))
    s = np.exp(np.asarray(ls, float))
    alpha = (c / s) ** 2
    beta = c / s ** 2
    return alpha, beta


def _from_z(z, m):
    """Unconstrained vector -> (weights, alpha, beta) of an m-component mixture."""
    if m == 1:
        a, b = _comp_from_z(z[0], z[1])
        return np.array([1.0]), np.array([float(a)]), np.array([float(b)])
    alr = z[:m - 1]
    lc = z[m - 1:2 * m - 1]
    ls = z[2 * m - 1:3 * m - 1]
    e = np.exp(np.concatenate([alr, [0.0]]))
    w = e / e.sum()
    a, b = _comp_from_z(lc, ls)
    return w, np.asarray(a, float), np.asarray(b, float)


# --------------------------------------------------------------------------
#  result objects
# --------------------------------------------------------------------------
@dataclass
class Shower:
    """One sampled longitudinal Cherenkov profile."""
    species: str
    pid: int
    energy: float                 # GeV
    z: np.ndarray                 # depth-bin centers [cm]
    photons: np.ndarray           # Cherenkov photons per bin
    m: int                        # number of gamma components drawn
    weights: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    beta: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    N: float = 0.0                # total sampled yield (amplitude)

    @property
    def z_m(self):
        """Depth-bin centers in meters."""
        return self.z / 100.0

    @property
    def integral(self):
        return float(_trapz(self.photons, self.z))

    def __repr__(self):
        return (f"Shower({self.species} @ {self.energy:.0f} GeV, m={self.m}, "
                f"N={self.N:.3g}, peak={self.photons.max():.3g})")


@dataclass
class Event:
    """A set of final-state hadron showers plus their composite (summed) profile."""
    showers: list
    z: np.ndarray                 # common depth grid [cm]
    composite: np.ndarray         # sum of the showers on z

    @property
    def z_m(self):
        return self.z / 100.0

    def __repr__(self):
        return f"Event({len(self.showers)} hadrons, peak={self.composite.max():.3g})"


# --------------------------------------------------------------------------
#  the model
# --------------------------------------------------------------------------
class HadronicShowerModel:
    """Loads the portable .npz and samples showers. Rebuilds the smooth
    energy-interpolation with cubic splines over the tabulated grid."""

    def __init__(self, per_pid, meta):
        self._pid = per_pid            # pid -> dict of rebuilt curves
        self.meta = meta
        self.Kmax = int(meta.get("Kmax", 3))

    # ---- construction ----
    @classmethod
    def load(cls, path=None):
        path = path or DEFAULT_MODEL
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"shower model not found at {path!r}. Build it with "
                "convert_model_to_npz.py and pass its path to load_model().")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["__meta__"]))
        per_pid = {}
        for pid_str, sp in meta["species"].items():
            pid = int(pid_str)
            logE = d[f"{pid}|logE"]
            entry = {
                "name": sp["name"],
                "logE_range": tuple(sp["logE_range"]),
                "z": d[f"{pid}|z"],
                "pm": CubicSpline(logE, d[f"{pid}|pm"], axis=0, extrapolate=True),
                "logN": CubicSpline(logE, d[f"{pid}|logN"], extrapolate=True),
                "logNsig": CubicSpline(logE, d[f"{pid}|logNsig"], extrapolate=True),
                "logE_lo": float(logE[0]),
                "logE_hi": float(logE[-1]),
                "m": {},
            }
            for m in sp["m_available"]:
                entry["m"][int(m)] = {
                    "mean": CubicSpline(logE, d[f"{pid}|m{m}|mean"], axis=0,
                                        extrapolate=True),
                    "cov": CubicSpline(logE, d[f"{pid}|m{m}|cov"], axis=0,
                                       extrapolate=True),
                }
            per_pid[pid] = entry
        return cls(per_pid, meta)

    # ---- introspection ----
    def species(self):
        """List of species names the model knows."""
        return [self._pid[p]["name"] for p in self._pid]

    def has(self, species):
        try:
            return self._resolve(species)[0] in self._pid
        except KeyError:
            return False

    def energy_range(self, species):
        """(E_min, E_max) in GeV over which this species was trained."""
        pid, _ = self._resolve(species)
        lo, hi = self._pid[pid]["logE_range"]
        return 10.0 ** lo, 10.0 ** hi

    def z_centers(self, species):
        pid, _ = self._resolve(species)
        return self._pid[pid]["z"]

    # ---- internals ----
    @staticmethod
    def _resolve(species):
        """Accept a name ('pip') or a PDG code (211) -> (pid, name)."""
        if isinstance(species, str):
            if species not in NAME_TO_PID:
                raise KeyError(f"unknown species name {species!r}")
            pid = NAME_TO_PID[species]
            return pid, species
        pid = int(species)
        if pid not in PID_TO_NAME:
            raise KeyError(f"unknown PDG code {pid} (K0=311 -> map to KS/KL first)")
        return pid, PID_TO_NAME[pid]

    def _lq(self, entry, E):
        """log10(E) clamped to the trained range, with a one-time out-of-range warn."""
        lq = np.log10(float(E))
        if lq < entry["logE_lo"] - 1e-9 or lq > entry["logE_hi"] + 1e-9:
            import warnings
            warnings.warn(
                f"E={E:.3g} GeV is outside the trained range "
                f"[{10**entry['logE_lo']:.3g}, {10**entry['logE_hi']:.3g}] GeV "
                f"for {entry['name']}; clamping to the edge.", stacklevel=3)
            lq = min(max(lq, entry["logE_lo"]), entry["logE_hi"])
        return lq

    def _p_m(self, entry, lq):
        p = np.clip(np.asarray(entry["pm"](lq), float), 0.0, None)
        s = p.sum()
        return p / s if s > 0 else np.ones_like(p) / len(p)

    def _mean_cov(self, entry, m, lq):
        mm = entry["m"].get(m)
        if mm is None:
            return None
        mean = np.asarray(mm["mean"](lq), float)
        cov = np.asarray(mm["cov"](lq), float)
        cov = 0.5 * (cov + cov.T)                          # symmetrize
        w, V = np.linalg.eigh(cov)                          # clip to PSD
        w = np.clip(w, _COV_RIDGE, None)
        cov = (V * w) @ V.T
        return mean, cov

    # ---- sampling ----
    def sample(self, species, E, rng=None, x=None, N=None):
        """Sample ONE shower. Returns a Shower. `x` overrides the depth grid [cm];
        `N` fixes the total yield (default: sampled from the yield distribution)."""
        rng = rng or np.random.default_rng()
        pid, name = self._resolve(species)
        if pid not in self._pid:
            raise KeyError(f"{name!r} is not in this model")
        entry = self._pid[pid]
        lq = self._lq(entry, E)
        x = entry["z"] if x is None else np.asarray(x, float)

        # pick m among those with fitted params, weighted by p(m)
        pm = self._p_m(entry, lq)
        avail = [m for m in range(1, self.Kmax + 1) if m in entry["m"]]
        p = np.array([pm[m - 1] for m in avail], float)
        p = p / p.sum()
        m = int(rng.choice(avail, p=p))

        mean, cov = self._mean_cov(entry, m, lq)
        z = rng.multivariate_normal(mean, cov)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z = np.clip(z, mean - _CLIP_SIGMA * sd, mean + _CLIP_SIGMA * sd)
        w, alpha, beta = _from_z(z, m)

        if N is None:
            N = float(np.exp(entry["logN"](lq)))
            s = float(max(entry["logNsig"](lq), 0.0))
            if s > 0:
                N *= float(np.exp(rng.normal(0.0, s)))
        photons = N * np.sum([w[i] * _kernel(x, alpha[i], beta[i])
                              for i in range(m)], axis=0)
        return Shower(species=name, pid=pid, energy=float(E), z=x, photons=photons,
                      m=m, weights=w, alpha=alpha, beta=beta, N=float(N))

    def sample_many(self, species, E, n, rng=None, seed=None):
        """n independent showers for one (species, energy). Returns list[Shower]."""
        if rng is None:
            rng = np.random.default_rng(seed)
        return [self.sample(species, E, rng) for _ in range(int(n))]

    def sample_event(self, final_state, rng=None, seed=None, composite=True, x=None):
        """Sample one shower per final-state hadron and (optionally) their composite.

        final_state : path to a final-state CSV, or a list of (species, E_GeV).
        Returns an Event. All showers share one depth grid so the composite is
        an exact bin-by-bin sum."""
        if rng is None:
            rng = np.random.default_rng(seed)
        if isinstance(final_state, str):
            final_state = load_final_state(final_state)
        # common depth grid: first known species' grid (all G4 species share binning)
        if x is None:
            for sp, _E in final_state:
                if self.has(sp):
                    x = self.z_centers(sp)
                    break
            if x is None:
                raise ValueError("no known species in the final state")
        x = np.asarray(x, float)

        showers, total = [], np.zeros_like(x)
        for sp, E in final_state:
            if not self.has(sp):
                continue
            s = self.sample(sp, E, rng, x=x)
            showers.append(s)
            if composite:
                total = total + s.photons
        if not showers:
            raise ValueError("no known species in the final state")
        return Event(showers=showers, z=x,
                     composite=(total if composite else None))


def load_model(path=None):
    """Load a portable shower model .npz. See HadronicShowerModel.load."""
    return HadronicShowerModel.load(path)


# --------------------------------------------------------------------------
#  final-state readers
# --------------------------------------------------------------------------
def load_final_state(path, event=None):
    """Read a final-state CSV -> list of (species, E_GeV).

    Columns: `species,E_GeV` or `event,species,E_GeV`. When an `event` column is
    present and `event` is None, the first event is used; pass `event=<i>` to
    select one. Lines beginning with '#' are ignored."""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None and any(c.isalpha() for c in parts[-1]):
                header = [p.lower() for p in parts]           # this is the header row
                continue
            rows.append(parts)
    if not rows:
        return []
    has_event = header is not None and header[0] in ("event", "ev", "idx")
    out = []
    for parts in rows:
        if has_event:
            ev, sp, E = int(float(parts[0])), parts[1], float(parts[2])
        else:
            sp, E = parts[0], float(parts[1])
            ev = 0
        out.append((ev, sp, E))
    if has_event:
        if event is None:
            event = out[0][0]
        return [(sp, E) for ev, sp, E in out if ev == event]
    return [(sp, E) for ev, sp, E in out]


def load_final_state_h5(path, enu_gev, event, rng=None):
    """One DIS event's hadronic secondaries from pythia_dis_secondaries.h5 ->
    list of (species, E_GeV). Needs h5py. K0/K0bar (311) -> KS/KL 50/50."""
    import h5py
    rng = rng or np.random.default_rng()
    key = f"E_nu_{enu_gev:.0e}"
    with h5py.File(path, "r") as f:
        if key not in f:
            raise KeyError(f"{key} not in {path}; groups = {list(f.keys())}")
        pids = np.asarray(f[key]["top20_pids"][event])
        ens = np.asarray(f[key]["top20_energies"][event])
    fs = []
    for pid, E in zip(pids, ens):
        pid = int(pid)
        if pid == 0 or not np.isfinite(E) or E <= 0:
            continue
        name = PID_TO_NAME.get(pid)
        if name is None and abs(pid) == 311:
            name = "KS" if rng.random() < 0.5 else "KL"
        if name is None:
            continue
        fs.append((name, float(E)))
    return fs


# --------------------------------------------------------------------------
#  plotting (matplotlib imported lazily so the sampler has no hard GUI dep)
# --------------------------------------------------------------------------
def _latex(name):
    return SPECIES_LATEX.get(name, name)


def _new_ax(ax, figsize=(8.4, 5.2)):
    import matplotlib.pyplot as plt
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(fig, ax, save, show):
    import matplotlib.pyplot as plt
    ax.set_xlabel("depth  z  [m]")
    ax.set_ylabel("Cherenkov photons / bin")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print("wrote", save)
    if show:
        plt.show()
    return fig, ax


def plot_many(model, species, E, n=100, rng=None, seed=0, ax=None,
              save=None, show=False, color="#4c78a8"):
    """MODE 1 -- overplot n sampled showers of ONE species + their mean."""
    showers = model.sample_many(species, E, n, rng=rng, seed=seed)
    fig, ax = _new_ax(ax)
    P = np.array([s.photons for s in showers])
    z_m = showers[0].z_m
    for p in P:
        ax.plot(z_m, p, color=color, lw=0.5, alpha=0.5)
    ax.set_xlim(0, 20)
    ax.plot(z_m, P.mean(0), color="#c0392b", lw=2.0, label="mean")
    ax.set_title(f"{n} sampled {_latex(showers[0].species)} showers at {E/1000:.0f} TeV")
    ax.legend()
    return _finish(fig, ax, save, show)


def plot_species(model, items, n=1, rng=None, seed=0, ax=None,
                 save=None, show=False, mean=False):
    """MODES 2 & 3 -- one shower (n=1) or n showers each for several species.

    items : list of (species, E_GeV)."""
    import matplotlib.pyplot as plt
    if rng is None:
        rng = np.random.default_rng(seed)
    fig, ax = _new_ax(ax)
    cmap = plt.get_cmap("tab10")
    for i, (sp, E) in enumerate(items):
        if not model.has(sp):
            print(f"  skip {sp!r}: not in the model")
            continue
        col = cmap(i % 10)
        showers = [model.sample(sp, E, rng) for _ in range(int(n))]
        z_m = showers[0].z_m
        lab = f"{_latex(showers[0].species)}  {E/1000:.0f} TeV"
        if n == 1:
            ax.plot(z_m, showers[0].photons, lw=1.6, color=col, label=lab)
        else:
            for s in showers:
                ax.plot(z_m, s.photons, lw=0.5, color=col, alpha=0.25)
            ref = np.mean([s.photons for s in showers], axis=0) if mean else showers[0].photons
            ax.plot(z_m, ref, lw=2.0, color=col,
                    label=lab + ("  (mean)" if mean else ""))
    ttl = "Sampled showers" if n == 1 else f"{n} sampled showers per species"
    ax.set_xlim(0, 20)
    ax.set_title(ttl)
    ax.legend(fontsize=8, ncol=2)
    return _finish(fig, ax, save, show)


def plot_event(model, final_state, event=None, composite=True, rng=None, seed=1,
               ax=None, save=None, show=False):
    """MODE 4 -- one shower per final-state hadron; overplot the composite (sum).

    final_state : CSV path or list of (species, E_GeV)."""
    if isinstance(final_state, str):
        final_state = load_final_state(final_state, event=event)
    ev = model.sample_event(final_state, rng=rng, seed=seed, composite=composite)
    fig, ax = _new_ax(ax)
    for s in ev.showers:
        ax.plot(ev.z_m, s.photons, lw=1.2, alpha=0.85,
                label=f"{_latex(s.species)} {s.energy:.0f} GeV (m={s.m})")
    if composite:
        ax.plot(ev.z_m, ev.composite, color="k", lw=2.6, label="composite (sum)")
    ax.set_xlim(0, 20)
    ax.set_title("Sampled hadronic shower of a final state")
    ax.legend(fontsize=8, ncol=2)
    return _finish(fig, ax, save, show)
