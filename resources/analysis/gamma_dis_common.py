"""
gamma_dis_common.py -- shared helpers for the GammaDISFromSpline validation plots.

  * plot style (matches the L2/KS comparison plots: bold 15-pt titles, dotted
    grids, 11-pt legends, the same colour scheme)
  * loading the .npz samples written by gamma_dis_samples.py
  * the spline "truth": the CSMS dsigma/dx dy evaluated on a fine (log x, log y)
    grid, normalised, and binned into any variable (x, y, Q^2) or pair of them.
    Grids are cached in theory_cache.npz because evaluating the spline is slow.
"""
import os
import numpy as np

SIREN_ROOT = os.environ.get("SIREN_ROOT", os.path.expanduser("~/SIREN"))
ELECTRON_MASS = 0.51099895e-3   # GeV; replaced by SIREN's constant when siren is importable
GAMMASUB = -2000002200          # ParticleType::GammaSubshower

# PDG rest masses [GeV], same table as rest_mass_GeV() in GammaDISFromSpline.cxx.
# The shower model / G4 library are labelled by KINETIC energy; SIREN stores TOTAL.
REST_MASS = {11: 0.51099895e-3, 22: 0.0, 111: 0.1349768, 211: 0.13957039,
             130: 0.497611, 310: 0.497611, 311: 0.497611, 321: 0.493677,
             2212: 0.93827208816, 2112: 0.93956542052}


def e_kin(pid, E):
    """Kinetic energy for the shower model from a total energy (same rule as the C++)."""
    return E - REST_MASS.get(abs(int(pid)), 0.0)


COLORS = {"gamma": "#F08070", "dis": "#4682B4", "theory": "#222222",
          "floor": "#444444", "test": "#F08070", "ref": "#4682B4"}
LABELS = {"gamma": "GammaDISFromSpline", "dis": "DISFromSpline",
          "theory": "Spline (CSMS)"}


# --------------------------------------------------------------------------
#  style
# --------------------------------------------------------------------------
def style(ax):
    ax.grid(True, which="major", ls=":", lw=0.9, color="#bbbbbb", alpha=0.7)
    ax.tick_params(axis="both", which="major", labelsize=12, length=6)


def title(ax, text):
    ax.set_title(text, fontsize=15, fontweight="bold", pad=10)


def legend(ax, loc="best"):
    ax.legend(fontsize=11, framealpha=0.92, loc=loc)


def elabel(E):
    """1e4 -> '10 TeV', 3e2 -> '300 GeV', 1e6 -> '1 PeV'."""
    E = float(E)
    for unit, scale in (("PeV", 1e6), ("TeV", 1e3), ("GeV", 1.0)):
        if E >= scale:
            return f"{E / scale:g} {unit}"
    return f"{E:g} GeV"


def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


# --------------------------------------------------------------------------
#  samples
# --------------------------------------------------------------------------
def load_sample(path):
    """Dict of arrays from gamma_dis_samples.py. Ragged per-event lists are
    stored flat with an offsets array (had_off, sub_off)."""
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def q2_of(s):
    """Q^2 = 2 M E x y for every event in a sample (same formula as the sampler)."""
    return 2.0 * s["target_mass"] * s["E_nu"] * s["x"] * s["y"]


def energy_components(E, w=None, emin=None, emax=None, nbins=16):
    """Split events into log-energy bins -> list of (E_center, summed weight).
    Used to build the energy-averaged truth for range samples."""
    E = np.asarray(E, float)
    w = np.ones_like(E) if w is None else np.asarray(w, float)
    emin = E.min() if emin is None else emin
    emax = E.max() if emax is None else emax
    edges = np.logspace(np.log10(emin), np.log10(emax), nbins + 1)
    idx = np.clip(np.digitize(E, edges) - 1, 0, nbins - 1)
    out = []
    for b in range(nbins):
        sw = w[idx == b].sum()
        if sw > 0:
            out.append((float(np.sqrt(edges[b] * edges[b + 1])), float(sw)))
    return out


# --------------------------------------------------------------------------
#  spline truth
# --------------------------------------------------------------------------
class SplineTruth:
    """Normalised CSMS CC nu_e dsigma/dx dy on a fine grid.

    For one energy E, cell weights W[i,j] = dsigma/dxdy(E,x_i,y_j) * dx_i * dy_j,
    normalised to sum 1, i.e. the probability that an interaction lands in that
    cell. Any histogram's expected bin probabilities are sums of W over cells.
    The spline already returns 0 below the Q^2 / W^2 cuts, outside the table, and
    in kinematically forbidden regions -- exactly the region the sampler uses.
    """

    GRID_VERSION = 2   # bump when the grid definition changes (invalidates the cache)

    def __init__(self, cache=None, nx=300, ny=250, xmin=1e-7, ymin=1e-6, sub=5):
        # x: log-spaced. y: log-spaced below 0.1, linear above, so that cells near
        # y ~ 1 stay much narrower than the (linear) y histogram bins.
        self.xe = np.logspace(np.log10(xmin), 0.0, nx + 1)
        n_lin = ny // 2
        self.ye = np.concatenate([np.logspace(np.log10(ymin), -1.0, ny - n_lin + 1),
                                  np.linspace(0.1, 1.0, n_lin + 1)[1:]])
        self.xc = np.sqrt(self.xe[1:] * self.xe[:-1])
        self.yc = np.sqrt(self.ye[1:] * self.ye[:-1])
        self.dx = np.diff(self.xe)
        self.dy = np.diff(self.ye)
        self.shape = (nx, ny)
        self.sub = sub
        self.cache_path = cache
        self._W = {}
        self._dis = None
        self.lepton_mass = ELECTRON_MASS
        if cache and os.path.exists(cache):
            with np.load(cache) as d:
                ok = (tuple(d["shape"]) == self.shape and "version" in d.files
                      and int(d["version"]) == self.GRID_VERSION)
                if ok:
                    for k in d.files:
                        if k.startswith("E_"):
                            self._W[float(k[2:])] = d[k]

    def _xs(self):
        if self._dis is None:
            import siren
            NUE = siren.dataclasses.Particle.ParticleType.NuE
            procs, _ = siren.utilities.load_processes(
                "CSMSDISSplines", primary_types=[NUE],
                target_types=[siren.dataclasses.Particle.ParticleType.Nucleon],
                isoscalar=True, process_types=["CC"])
            self._dis = procs[NUE][0]
            try:
                self.lepton_mass = siren.utilities.Constants.electronMass
            except AttributeError:
                pass
        return self._dis

    def weights(self, E):
        """Normalised cell probabilities at energy E (computed once, cached)."""
        key = float(f"{float(E):.6e}")
        if key not in self._W:
            xs, m = self._xs(), self.lepton_mass
            nan = float("nan")
            f = np.array([[xs.DifferentialCrossSection(key, x, y, m, nan)
                           for y in self.yc] for x in self.xc])
            W = np.nan_to_num(f) * self.dx[:, None] * self.dy[None, :]
            tot = W.sum()
            if tot <= 0:
                raise RuntimeError(f"spline is zero everywhere at E={key} GeV "
                                   "(outside the table's energy range?)")
            self._W[key] = W / tot
            self._save()
        return self._W[key]

    def _save(self):
        if not self.cache_path:
            return
        arrs = {f"E_{k:.6e}": v for k, v in self._W.items()}
        np.savez(self.cache_path, shape=np.array(self.shape),
                 version=np.array(self.GRID_VERSION), **arrs)

    def grid_vars(self, E, M):
        """Variables at the cell centres (used for means)."""
        X, Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        return {"x": X, "y": Y, "Q2": 2.0 * M * E * X * Y}

    def _subsampled(self, E, M):
        """Each cell split into sub x sub points (density taken uniform inside the
        cell) -> flat arrays of x, y, Q2 and probability. Used for Q^2 and 2D maps
        so that no histogram bin is starved by coarse cells."""
        k = self.sub
        f = (np.arange(k) + 0.5) / k
        xs = (self.xe[:-1, None] + f[None, :] * self.dx[:, None]).ravel()
        ys = (self.ye[:-1, None] + f[None, :] * self.dy[:, None]).ravel()
        Wsub = np.repeat(np.repeat(self.weights(E), k, 0), k, 1) / k ** 2
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return {"x": X.ravel(), "y": Y.ravel(), "Q2": (2.0 * M * E * X * Y).ravel()}, Wsub.ravel()

    @staticmethod
    def _rebin(edges, mass, bins):
        """Exact re-binning of piecewise-uniform cell masses onto new bins."""
        cum = np.concatenate([[0.0], np.cumsum(mass)])
        return np.diff(np.interp(bins, edges, cum))

    def expected_1d(self, var, bins, comps, M):
        """Expected probability per bin of `var` for a mixture of energies
        comps = [(E, weight), ...]."""
        tot = np.zeros(len(bins) - 1)
        wsum = 0.0
        for E, w in comps:
            if var == "x":
                p = self._rebin(self.xe, self.weights(E).sum(axis=1), bins)
            elif var == "y":
                p = self._rebin(self.ye, self.weights(E).sum(axis=0), bins)
            else:
                v, W = self._subsampled(E, M)
                p = np.histogram(v[var], bins, weights=W)[0]
            tot += w * p
            wsum += w
        return tot / wsum

    def expected_2d(self, va, vb, ba, bb, comps, M):
        tot = np.zeros((len(ba) - 1, len(bb) - 1))
        wsum = 0.0
        for E, w in comps:
            v, W = self._subsampled(E, M)
            tot += w * np.histogram2d(v[va], v[vb], [ba, bb], weights=W)[0]
            wsum += w
        return tot / wsum

    def mean(self, var, E, M):
        """<var> at a single energy (e.g. <y>, <Q^2>)."""
        return float(np.sum(self.weights(E) * self.grid_vars(E, M)[var]))
