"""
hadronic_showers.py  --  fast generative hadronic shower sampler for SIREN
==========================================================================
Sample the longitudinal Cherenkov profile of a hadronic shower in ice from a
model calibrated to Geant4 (NO Geant4 needed at run time). Each shower's profile
is a sum of gamma kernels whose parameters are drawn from energy-interpolated
distributions, so event-to-event fluctuations are preserved.

The model data is a portable .npz.

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
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# --------------------------------------------------------------------------
#  hadron types (must match geant4_shower/.../shower_gamma_model.SPECIES)
# --------------------------------------------------------------------------
SPECIES = [
    (111, "pi0"), (211, "pip"), (-211, "pim"),
    (321, "Kp"), (-321, "Km"), (310, "KS"), (130, "KL"),
    (2212, "p"), (2112, "n"),
    (11, "em"), (-11, "ep"),        # electron / positron (EM showers)
]
NAME_TO_PID = {name: pid for pid, name in SPECIES}
PID_TO_NAME = {pid: name for pid, name in SPECIES}
SPECIES_LATEX = {
    "pip": r"$\pi^{+}$", "pim": r"$\pi^{-}$", "pi0": r"$\pi^{0}$",
    "Kp": r"$K^{+}$", "Km": r"$K^{-}$", "KS": r"$K^{0}_{S}$", "KL": r"$K^{0}_{L}$",
    "p": r"$p$", "n": r"$n$", "em": r"$e^{-}$", "ep": r"$e^{+}$",
}

# default model location (under resources/)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    _HERE, "..", "resources", "showers",
    "GammaShowerModel-v1.0", "shower_model.npz",
)

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


class _Const:
    """Constant curve: returns the stored value for any energy. Used for a
    species trained at a single energy (e.g. electron at just 1 TeV), where a
    spline over one point is impossible."""
    def __init__(self, value):
        self.value = np.asarray(value, float)

    def __call__(self, q):
        return self.value


def _make_curve(logE, y):
    """CubicSpline over the energy grid, or a constant if the grid has <2
    distinct (strictly increasing) points -- the single-energy case."""
    logE = np.asarray(logE, float)
    y = np.asarray(y, float)
    if logE.size < 2 or not np.all(np.diff(logE) > 0):
        return _Const(y[0])
    return CubicSpline(logE, y, axis=0, extrapolate=True)


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
                "pm": _make_curve(logE, d[f"{pid}|pm"]),
                "logN": _make_curve(logE, d[f"{pid}|logN"]),
                "logNsig": _make_curve(logE, d[f"{pid}|logNsig"]),
                "logE_lo": float(logE[0]),
                "logE_hi": float(logE[-1]),
                "m": {},
            }
            for m in sp["m_available"]:
                entry["m"][int(m)] = {
                    "mean": _make_curve(logE, d[f"{pid}|m{m}|mean"]),
                    "cov": _make_curve(logE, d[f"{pid}|m{m}|cov"]),
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

    def yield_mean(self, species, E):
        """Mean total Cherenkov yield (amplitude) at energy E [GeV]."""
        pid, _ = self._resolve(species)
        entry = self._pid[pid]
        return float(np.exp(entry["logN"](self._lq(entry, E))))

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
                f"for {entry['name']}; holding shape at the edge and scaling "
                f"yield proportional to E.", stacklevel=3)
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
            # Outside the trained range lq is clamped to the nearest bound: we hold
            # the shape there but scale the yield proportional to the true energy
            # (yield ~ E), the same rescaling the G4 reference uses. In-range this
            # factor is exactly 1 (10**lq == E), so it only affects out-of-range E.
            N *= float(E) / (10.0 ** lq)
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
#  plotting (matplotlib imported)
# --------------------------------------------------------------------------
def _latex(name):
    return SPECIES_LATEX.get(name, name)


def _elabel(E):
    """Energy label: 1000 -> '1 TeV', 300 -> '300 GeV'."""
    E = float(E)
    return f"{E / 1000:g} TeV" if E >= 1000 else f"{E:g} GeV"


# profiles are always plotted as the unit-area density (integral over x[cm] = 1)
XLABEL = r"$x$  [cm]"
YLABEL = r"$\hat{\ell}_{\mathrm{tot}}^{-1}\,\mathrm{d}\hat{\ell}/\mathrm{d}x$  [1/cm]"

def _norm_xy(shower):
    """Depth [cm] and unit-area density [1/cm] for one shower."""
    x = shower.z
    y = np.asarray(shower.photons, float)
    area = _trapz(y, x)
    if area > 0:
        y = y / area
    return x, y


def _new_ax(ax, figsize=(8.4, 5.2)):
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _finish(fig, ax, save, show):
    ax.set_xlabel(XLABEL, fontsize=16)
    ax.set_ylabel(YLABEL, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.title.set_fontsize(18)
    leg = ax.get_legend()

    # Scientific notation on y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
        print("wrote", save)
    if show:
        plt.show()
    return fig, ax


def plot_many(model, species, E, n=100, rng=None, seed=0, ax=None,
              save=None, show=False, color="#4c78a8", xlim=(0, 1700)):
    """Overplot n sampled showers of ONE species + their mean (normalized density)."""
    showers = model.sample_many(species, E, n, rng=rng, seed=seed)
    fig, ax = _new_ax(ax)
    xy = [_norm_xy(s) for s in showers]
    x0 = xy[0][0]
    Y = np.array([y for _x, y in xy])
    for y in Y:
        ax.plot(x0, y, color=color, lw=0.5, alpha=0.5)
    ax.plot(x0, Y.mean(0), color="#c0392b", lw=2.0, label="mean")
    ax.set_xlim(*xlim)
    ax.set_title(f"{n} sampled {_latex(showers[0].species)} showers at {E/1000:.0f} TeV")
    ax.legend()
    return _finish(fig, ax, save, show)


def plot_species(model, items, n=1, rng=None, seed=0, ax=None,
                 save=None, show=False, mean=False, xlim=(0, 1700)):
    """One shower (n=1) or n showers each for several species (normalized density).

    items : list of (species, E_GeV)."""
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
        xy = [_norm_xy(s) for s in showers]
        x0 = xy[0][0]
        lab = f"{_latex(showers[0].species)}  {E/1000:.0f} TeV"
        if n == 1:
            ax.plot(x0, xy[0][1], lw=1.6, color=col, label=lab)
        else:
            for _x, y in xy:
                ax.plot(x0, y, lw=0.5, color=col, alpha=0.25)
            ref = np.mean([y for _x, y in xy], axis=0) if mean else xy[0][1]
            ax.plot(x0, ref, lw=2.0, color=col,
                    label=lab + ("  (mean)" if mean else ""))
    ax.set_xlim(*xlim)
    ax.set_title("Sampled showers" if n == 1 else f"{n} sampled showers per species")
    ax.legend()
    return _finish(fig, ax, save, show)


def plot_event(model, final_state, event=None, composite=True, rng=None, seed=1,
               ax=None, save=None, show=False, xlim=(0, 1700)):
    """One shower per final-state hadron + composite (sum), normalized density.
    Every curve is divided by one shared scale (the composite's integral) so the
    hadrons still add up to the composite and the composite integrates to 1.

    final_state : CSV path or list of (species, E_GeV)."""
    if isinstance(final_state, str):
        final_state = load_final_state(final_state, event=event)
    ev = model.sample_event(final_state, rng=rng, seed=seed, composite=composite)
    fig, ax = _new_ax(ax)
    xg = ev.z
    if composite and ev.composite is not None:
        area = _trapz(ev.composite, xg)
    else:
        area = sum(_trapz(s.photons, xg) for s in ev.showers)
    scale = area if area > 0 else 1.0
    for s in ev.showers:
        ax.plot(xg, s.photons / scale, lw=1.2, alpha=0.85,
                label=f"{_latex(s.species)} {s.energy:.0f} GeV (m={s.m})")
    if composite:
        ax.plot(xg, ev.composite / scale, color="k", lw=2.6, label="composite (sum)")
    ax.set_xlim(*xlim)
    ax.set_title("Sampled hadronic shower of a final state")
    ax.legend()
    return _finish(fig, ax, save, show)


def plot_overlay(model, items, n=100, rng=None, seed=0, colors=None,
                 lw=0.5, alpha=0.5, xlim=(0, 1700), ax=None, save=None, show=False):
    """Overplot n sampled showers for each (species, E) item, colored per item --
    the e- vs pi+ style comparison plot (normalized density).

    items : list of (species, E_GeV), e.g. [("em", 1000), ("pip", 1000)].
    Species not (yet) in the model are skipped with a note."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if colors is None:
        colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, ax = _new_ax(ax, figsize=(8, 6))
    for i, (sp, E) in enumerate(items):
        if not model.has(sp):
            print(f"  skip {sp!r}: not in the model")
            continue
        col = colors[i % len(colors)]
        first = True
        for _ in range(int(n)):
            x, y = _norm_xy(model.sample(sp, E, rng))
            ax.plot(x, y, color=col, lw=lw, alpha=alpha,
                    label=(f"{_elabel(E)} {_latex(sp)} ({n} runs)" if first else None))
            first = False
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend()
    return _finish(fig, ax, save, show)


# ---------------------------------------------------------------------------
#  whole-event composite profile plots
# ---------------------------------------------------------------------------
def _norm_curve(z_cm, y):
    """Unit-area density [1/cm] of a composite profile."""
    y = np.asarray(y, float)
    area = _trapz(y, z_cm)
    return y / area if area > 0 else y


def top_k_final_state(pids, energies, e_had, top_k=10, rng=None):
    """Energy-conserving final state: the top_k most energetic model-known hadrons
    plus ONE pi0 carrying e_had - sum(top_k). By construction the total energy of
    the returned hadrons equals e_had exactly."""
    rng = rng or np.random.default_rng()
    hadrons = []
    for pid, e in zip(pids, energies):
        pid = int(pid)
        if pid == 0 or not np.isfinite(e) or e <= 0:
            continue
        name = PID_TO_NAME.get(pid)
        if name is None and abs(pid) == 311:
            name = "KS" if rng.random() < 0.5 else "KL"
        if name in (None, "em", "ep"):            # skip non-hadrons / leptons
            continue
        hadrons.append((name, float(e)))
    hadrons.sort(key=lambda t: -t[1])
    top = hadrons[:top_k]
    rem = max(float(e_had) - sum(e for _, e in top), 0.0)
    fs = list(top)
    if rem > 0:
        fs.append(("pi0", rem))
    return fs


def _read_event_store(path):
    """Load the PYTHIA events store: a .npz (from pythia_events_to_npz.py) with
    arrays E_had, top20_pids, top20_energies. Numpy-only -- no h5py at run time."""
    if not str(path).endswith(".npz"):
        raise ValueError("event store must be the .npz built by pythia_events_to_npz.py")
    d = np.load(path, allow_pickle=False)
    return (np.asarray(d["E_had"], float),
            np.asarray(d["top20_pids"]),
            np.asarray(d["top20_energies"], float))


def event_bins(EH, n_bins=12):
    """Log-spaced E_had bin edges over the event store (same scheme as the L2 plot)."""
    lo, hi = max(float(EH.min()), 1.0), float(EH.max())
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)


def select_events(source, e_had=None, n=100, n_bins=12, top_k=10, rng=None):
    """Pick events from the store and build energy-conserving final states.

    You give a SPECIFIC energy `e_had`; we find the log-E_had bin it lands in and
    sample `n` events from that bin, widening into the nearest neighbouring bins
    only if the bin holds fewer than `n`. Returns (final_states, (Emin, Emax)) --
    the actual E_had range of the picked events, for labelling. With e_had=None a
    random `n` events are taken."""
    rng = rng or np.random.default_rng()
    EH, PD, EN = _read_event_store(source)
    if e_had is None:
        sel = rng.choice(len(EH), min(n, len(EH)), replace=False)
    else:
        edges = event_bins(EH, n_bins)
        which = np.clip(np.digitize(EH, edges) - 1, 0, n_bins - 1)
        b = int(np.clip(np.digitize([e_had], edges)[0] - 1, 0, n_bins - 1))
        pool = []
        for k in sorted(range(n_bins), key=lambda kk: abs(kk - b)):   # bin b, then neighbours
            pool.extend(np.where(which == k)[0].tolist())
            if len(pool) >= n:
                break
        pool = np.asarray(pool)
        sel = rng.choice(pool, min(n, len(pool)), replace=False)
    fss = [top_k_final_state(PD[i], EN[i], EH[i], top_k, rng) for i in sel]
    erange = (float(EH[sel].min()), float(EH[sel].max()))
    return fss, erange


def _erange_label(erange, requested=None):
    lo, hi = erange
    s = rf"$E_{{\mathrm{{had}}}}\in$[{lo:.0f}, {hi:.0f}] GeV"
    return s + (rf"  (req {requested:.0f})" if requested is not None else "")


def plot_event_fluctuation(model, source=None, e_had=None, final_state=None, n=100,
                           n_bins=12, top_k=10, rng=None, seed=0, color="#4c78a8",
                           xlim=(0, 1700), ax=None, save=None, show=False):
    """(a) Sample ONE event's composite n times -> the MODEL's own spread for a FIXED
    final state (a narrow band = sampler noise). Normalized.

    Provide either `final_state` directly, or `source` + `e_had`: one event is then
    drawn at random from the E_had bin containing e_had."""
    if rng is None:
        rng = np.random.default_rng(seed)
    if final_state is None:
        if source is None or e_had is None:
            raise ValueError("give final_state, or source and e_had")
        fss, _er = select_events(source, e_had=e_had, n=1, n_bins=n_bins,
                                 top_k=top_k, rng=rng)
        final_state = fss[0]
    elif isinstance(final_state, str):
        final_state = load_final_state(final_state)
    fig, ax = _new_ax(ax)
    Y, z = [], None
    for _ in range(int(n)):
        ev = model.sample_event(final_state, rng=rng)
        z = ev.z if z is None else z
        y = _norm_curve(ev.z, ev.composite)
        Y.append(y)
        ax.plot(ev.z, y, color=color, lw=0.4, alpha=0.25)
    ax.plot(z, np.mean(Y, axis=0), color="#c0392b", lw=2.4, label="mean")
    ax.set_xlim(*xlim)
    Etot = sum(E for _, E in final_state)
    ax.set_title(rf"{n} samples of one event  ($E_{{\mathrm{{had}}}}$ = {Etot:.0f} GeV)")
    ax.legend()
    return _finish(fig, ax, save, show)


def plot_events(model, events, rng=None, seed=0, color="#4c78a8", mean=True,
                xlim=(0, 1700), title=None, ax=None, save=None, show=False):
    """(b) One composite per event -> the PHYSICAL event-to-event spread. `events`
    is a list of final states (each [(species, E), ...]). Normalized."""
    if rng is None:
        rng = np.random.default_rng(seed)
    fig, ax = _new_ax(ax)
    Y, z = [], None
    for fs in events:
        if not fs:
            continue
        ev = model.sample_event(fs, rng=rng)
        z = ev.z if z is None else z
        y = _norm_curve(ev.z, ev.composite)
        Y.append(y)
        ax.plot(ev.z, y, color=color, lw=0.4, alpha=0.25)
    if mean and Y:
        ax.plot(z, np.mean(Y, axis=0), color="#c0392b", lw=2.4, label="mean")
        ax.legend()
    ax.set_xlim(*xlim)
    ax.set_title(title or f"{len(Y)} sampled event composites")
    return _finish(fig, ax, save, show)


def plot_events_at_energy(model, source, e_had, n=100, n_bins=12, top_k=10, seed=0,
                          xlim=(0, 1700), save=None, show=False):
    """(b) Sample one composite each for n events from the E_had bin containing
    `e_had` (widening to neighbours if sparse). `source` is the pythia events .npz.
    The plot title shows the actual E_had range of the events used."""
    rng = np.random.default_rng(seed)
    events, er = select_events(source, e_had=e_had, n=n, n_bins=n_bins,
                               top_k=top_k, rng=rng)
    title = f"{len(events)} event composites,  " + _erange_label(er, e_had)
    return plot_events(model, events, rng=rng, xlim=xlim, title=title,
                       save=save, show=show)
