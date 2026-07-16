"""
shower_gamma_model.py
=====================
Build a fast generative model of hadronic sub-shower Cherenkov profiles from a
Geant4 profile library, with NO Geant4 at runtime.

Three modules in one file:

  A. Fitting        : fit each G4 profile as a sum of m gamma kernels; choose m
                      by BIC.  Aggregate -> empirical  p(m | type, E)  and
                      p(w, alpha, beta | m, type, E).
  B. Interpolation  : spline the distribution parameters in log-E so they can be
                      evaluated at any energy.
  C. Sampling       : given (type, E), sample m, then (w, alpha, beta), and build
                      a profile  N * sum_i w_i * Gamma(x; alpha_i, beta_i).

(Interaction-length positioning is intentionally left out for now.)

Run
---
    # synthetic end-to-end check (no data needed)
    python shower_gamma_model.py --selftest

    # BUILD ONCE from real G4 files, in parallel, and SAVE the model:
    python shower_gamma_model.py --g4-dir ../output --n-jobs -1 \
                                 --save ../output/shower_model.pkl

    # LATER: load the saved model (instant) and validate / sample -- no refit:
    python shower_gamma_model.py --g4-dir ../output --load ../output/shower_model.pkl \
                                 --species pip --energy 1000

In code (this is what the event-assembly step will do):
    from shower_gamma_model import load_model, ShowerSampler
    samp = ShowerSampler(load_model("shower_model.pkl"))
    prof, info = samp.sample_profile(pid, E, rng)      # one secondary; sum over secondaries

Real G4 files are expected as  <g4-dir>/shower_<name>_E<E>GeV.h5  with datasets
  profiles (n_runs, n_bins), z_edges (n_bins+1,), N_total (n_runs,).
"""

import os
import re
import glob
import argparse
import numpy as np
from scipy.special import gammaln
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, interp1d

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.x renamed trapz

# ---------------------------------------------------------------------------
# species metadata (name used in filenames <-> PDG id)
# ---------------------------------------------------------------------------
SPECIES = [
    (111, "pi0"), (211, "pip"), (-211, "pim"),
    (321, "Kp"), (-321, "Km"), (310, "KS"), (130, "KL"),
    (2212, "p"), (2112, "n"),
]
NAME_TO_PID = {name: pid for pid, name in SPECIES}
PID_TO_NAME = {pid: name for pid, name in SPECIES}

KMAX_DEFAULT = 3
MIN_SAMPLES_FOR_COV = 12   # min runs at an (m, E) to estimate a covariance


# ===========================================================================
#  IO
# ===========================================================================
def load_g4_library(g4_dir):
    """Return {pid: {E_GeV: {'profiles','N_total','z_edges','z_centers'}}}."""
    files = sorted(glob.glob(os.path.join(g4_dir, "shower_*_E*GeV.h5")))
    if not files:
        raise FileNotFoundError(f"no shower_*_E*GeV.h5 in {g4_dir}")
    import h5py
    lib = {}
    for f in files:
        m = re.match(r"shower_(\w+)_E(\d+)GeV\.h5", os.path.basename(f))
        if not m:
            continue
        pid = NAME_TO_PID.get(m.group(1))
        if pid is None:
            continue
        E = float(m.group(2))
        with h5py.File(f, "r") as hf:
            prof = hf["profiles"][:].astype(float)
            z_edges = hf["z_edges"][:]
            N_total = hf["N_total"][:] if "N_total" in hf else prof.sum(axis=1)
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        lib.setdefault(pid, {})[E] = dict(profiles=prof, N_total=N_total,
                                          z_edges=z_edges, z_centers=zc)
    return lib


# ===========================================================================
#  MODULE A :  fitting a profile to a sum of gammas
# ===========================================================================
def _kernel(x, alpha, beta):
    """Unit-area gamma kernel:  beta^a x^(a-1) e^(-b x)/Gamma(a)  (fast, log-space)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    pos = x > 0
    lk = (alpha * np.log(beta) + (alpha - 1.0) * np.log(x[pos])
          - beta * x[pos] - gammaln(alpha))
    out[pos] = np.exp(lk)
    return out


def _mixture(x, *p):
    K = len(p) // 3
    y = np.zeros_like(x, dtype=float)
    for i in range(K):
        A, a, b = p[3 * i], p[3 * i + 1], p[3 * i + 2]
        y += A * _kernel(x, a, b)
    return y


def _sort_components(A, alpha, beta):
    mode = np.maximum(alpha - 1.0, 1e-6) / beta
    o = np.argsort(mode)
    return A[o], alpha[o], beta[o]


def _initial_guess(x, y, K, smooth=5):
    """Seed K components from the K strongest peaks (fallback: quantiles)."""
    ys = np.convolve(y, np.ones(smooth) / smooth, mode="same") if smooth > 1 else y
    pk = ys.max() if ys.max() > 0 else 1.0
    peaks, props = find_peaks(ys, prominence=0.03 * pk, distance=8)
    if len(peaks) >= K:
        top = peaks[np.argsort(ys[peaks])[::-1][:K]]
    else:  # not enough peaks: pad with data quantiles
        cdf = np.cumsum(np.maximum(y, 0));  cdf /= max(cdf[-1], 1e-12)
        qs = np.linspace(0.2, 0.8, K)
        top = np.array([np.searchsorted(cdf, q) for q in qs])
    top = np.clip(top, 1, len(x) - 1)
    p0, lo, hi = [], [], []
    a0 = 6.0
    for idx in np.sort(top):
        mode = max(x[idx], x[1])
        b0 = (a0 - 1.0) / mode
        A0 = max(ys[idx], pk * 0.05) / max(_kernel(mode, a0, b0), 1e-12)
        p0 += [A0, a0, b0]
        lo += [0.0, 1.0, 1e-5]
        hi += [np.inf, 300.0, 0.2]     # cap alpha & beta: no sub-bin "spike" components
    return p0, (lo, hi)


def fit_profile(x, y, Kmax=KMAX_DEFAULT):
    """
    Fit y(x) as sums of 1..Kmax gammas; pick K by BIC.
    Returns dict with m, w (fractional, sums to 1), alpha, beta, N (total yield),
    and the BIC/rss trace.
    """
    n = len(x)
    total = _trapz(y, x)
    best = None
    trace = {}
    for K in range(1, Kmax + 1):
        try:
            p0, bounds = _initial_guess(x, y, K)
            popt, _ = curve_fit(_mixture, x, y, p0=p0, bounds=bounds, maxfev=6000)
        except Exception:
            trace[K] = np.inf
            continue
        resid = y - _mixture(x, *popt)
        rss = float(np.sum(resid ** 2))
        k = 3 * K
        bic = n * np.log(max(rss, 1e-30) / n) + k * np.log(n)
        trace[K] = bic
        if best is None or bic < best["bic"]:
            A = np.array(popt[0::3]); al = np.array(popt[1::3]); be = np.array(popt[2::3])
            A, al, be = _sort_components(A, al, be)
            Asum = A.sum()
            best = dict(m=K, w=A / Asum if Asum > 0 else np.ones(K) / K,
                        alpha=al, beta=be, N=Asum if Asum > 0 else total, bic=bic)
    if best is None:                       # every fit failed -> 1 broad gamma
        best = dict(m=1, w=np.array([1.0]), alpha=np.array([4.0]),
                    beta=np.array([4.0 / max(_trapz(x * y, x) / max(total, 1e-9), 1.0)]),
                    N=total, bic=np.inf)
    best["bic_trace"] = trace
    return best


# ===========================================================================
#  MODULE A (aggregation) : empirical distributions per (type, E)
# ===========================================================================
def _to_z(w, alpha, beta):
    """Map (w, alpha, beta) of an m-component fit to an unconstrained vector."""
    m = len(alpha)
    la, lb = np.log(alpha), np.log(beta)
    if m == 1:
        return np.concatenate([la, lb])                 # dim 2
    w = np.clip(w, 1e-6, None); w = w / w.sum()
    alr = np.log(w[:-1]) - np.log(w[-1])                # dim m-1
    return np.concatenate([alr, la, lb])                # dim 3m-1


def _from_z(z, m):
    if m == 1:
        return np.array([1.0]), np.array([np.exp(z[0])]), np.array([np.exp(z[1])])
    alr = z[:m - 1]; la = z[m - 1:2 * m - 1]; lb = z[2 * m - 1:3 * m - 1]
    e = np.exp(np.concatenate([alr, [0.0]])); w = e / e.sum()
    return w, np.exp(la), np.exp(lb)


def build_distributions(library, Kmax=KMAX_DEFAULT, verbose=True, n_jobs=1):
    """
    Fit every run and aggregate.  Fitting is embarrassingly parallel; set
    n_jobs>1 (or -1 for all cores) to fit profiles across worker processes.
    Returns {pid: {E: {'p_m':(Kmax,), 'Z':{m: array(n_m, 3m-1)}, 'N_mean':float,
                       'z_centers':...}}}
    """
    pool = None
    if n_jobs != 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=(None if n_jobs < 0 else n_jobs))
    dists = {}
    try:
        for pid, edata in library.items():
            dists[pid] = {}
            for E, d in sorted(edata.items()):
                x = d["z_centers"]; profs = d["profiles"]
                valid = [y for y in profs if y.max() > 0]
                if pool is not None:
                    fits = pool.starmap(fit_profile, [(x, y, Kmax) for y in valid])
                else:
                    fits = [fit_profile(x, y, Kmax) for y in valid]
                counts = np.zeros(Kmax, dtype=int)
                Z = {m: [] for m in range(1, Kmax + 1)}
                Ns = []
                for fit in fits:
                    m = fit["m"]
                    counts[m - 1] += 1
                    Z[m].append(_to_z(fit["w"], fit["alpha"], fit["beta"]))
                    Ns.append(fit["N"])
                tot = counts.sum()
                # Yield stats come from the FITTED amplitudes (sum of A_i). These are
                # the units the sampler rebuilds in (N * unit-area kernels -> a bin sum
                # of N/binwidth), so N_mean must be in these units, NOT raw G4 N_total.
                # The log-spread still equals the G4 N_total spread (the bin-width
                # factor is constant and cancels), so the fluctuation is faithful.
                logN = np.log(np.maximum(Ns, 1e-9)) if Ns else np.array([0.0])
                dists[pid][E] = dict(
                    p_m=counts / max(tot, 1),
                    Z={m: np.array(v) for m, v in Z.items() if len(v) > 0},
                    N_mean=float(np.median(Ns)) if Ns else 0.0,   # median: robust central yield
                    N_logsigma=float(np.std(logN)),               # event-to-event yield spread
                    z_centers=x,
                )
                if verbose:
                    pm = np.round(dists[pid][E]["p_m"], 3)
                    print(f"  {PID_TO_NAME.get(pid,pid):4s} E={E:8.0f} GeV  "
                          f"n={tot:4d}  p(m)={pm}  <N>={dists[pid][E]['N_mean']:.3g}")
    finally:
        if pool is not None:
            pool.close(); pool.join()
    return dists


# ===========================================================================
#  MODULE B : interpolate distribution parameters in log-E
# ===========================================================================
class _Const:
    """Picklable constant callable (replaces a lambda so the model can be pickled)."""
    def __init__(self, v):
        self.v = float(v)

    def __call__(self, q):
        return np.full_like(np.asarray(q, float), self.v)


def _make_1d(xs, ys):
    """Return callable f(x): cubic if >=3 pts, linear if 2, constant if 1."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if len(xs) >= 3:
        return CubicSpline(xs, ys, extrapolate=True)
    if len(xs) == 2:
        return interp1d(xs, ys, kind="linear", fill_value="extrapolate")
    return _Const(ys[0])


class ShowerParamInterpolator:
    """Splines p(m), the (mean, cov) of the transformed params per m, and N."""

    def __init__(self, dists, Kmax=KMAX_DEFAULT, cov_ridge=1e-3):
        self.Kmax = Kmax
        self.cov_ridge = cov_ridge
        self.pid_models = {}
        for pid, edata in dists.items():
            Es = sorted(edata.keys())
            logE = np.log10(Es)
            zc = edata[Es[0]]["z_centers"]
            # p(m) vs logE
            pm_curves = [_make_1d(logE, [edata[E]["p_m"][k] for E in Es])
                         for k in range(Kmax)]
            # log N vs logE  (+ its spread, for yield fluctuation)
            logN = _make_1d(logE, [np.log(max(edata[E]["N_mean"], 1e-9)) for E in Es])
            logNsig = _make_1d(logE, [edata[E].get("N_logsigma", 0.0) for E in Es])
            # per-m mean & cov of transformed vector
            m_models = {}
            for m in range(1, Kmax + 1):
                dim = 2 if m == 1 else 3 * m - 1
                Es_m, means, covs = [], [], []
                for E in Es:
                    Z = edata[E]["Z"].get(m)
                    if Z is None or len(Z) < 2:
                        continue
                    Es_m.append(E)
                    means.append(Z.mean(axis=0))
                    if len(Z) >= MIN_SAMPLES_FOR_COV:
                        covs.append(np.cov(Z, rowvar=False).reshape(dim, dim))
                    else:                          # too few: diagonal from spread
                        covs.append(np.diag(np.var(Z, axis=0) + 1e-6))
                if not Es_m:
                    continue
                lE = np.log10(Es_m)
                mean_sp = [_make_1d(lE, [mu[i] for mu in means]) for i in range(dim)]
                cov_sp = [[_make_1d(lE, [C[i, j] for C in covs]) for j in range(dim)]
                          for i in range(dim)]
                m_models[m] = dict(dim=dim, mean_sp=mean_sp, cov_sp=cov_sp,
                                   logE_range=(lE.min(), lE.max()))
            self.pid_models[pid] = dict(pm=pm_curves, logN=logN, logNsig=logNsig,
                                        m_models=m_models, z_centers=zc,
                                        logE_range=(logE.min(), logE.max()))

    def p_m(self, pid, E):
        cur = self.pid_models[pid]["pm"]
        p = np.array([float(np.clip(c(np.log10(E)), 0, None)) for c in cur])
        s = p.sum()
        return p / s if s > 0 else np.ones(self.Kmax) / self.Kmax

    def yield_mean(self, pid, E):
        return float(np.exp(self.pid_models[pid]["logN"](np.log10(E))))

    def yield_logsigma(self, pid, E):
        sp = self.pid_models[pid].get("logNsig")
        return float(max(sp(np.log10(E)), 0.0)) if sp is not None else 0.0

    def mean_cov(self, pid, E, m):
        mm = self.pid_models[pid]["m_models"].get(m)
        if mm is None:
            return None
        lq = np.log10(E); dim = mm["dim"]
        mean = np.array([sp(lq) for sp in mm["mean_sp"]], float)
        cov = np.array([[mm["cov_sp"][i][j](lq) for j in range(dim)]
                        for i in range(dim)], float)
        cov = 0.5 * (cov + cov.T)                          # symmetrize
        w, V = np.linalg.eigh(cov)                          # clip to PSD
        w = np.clip(w, self.cov_ridge, None)
        cov = (V * w) @ V.T
        return mean, cov

    def z_centers(self, pid):
        return self.pid_models[pid]["z_centers"]


# ===========================================================================
#  MODULE C : sampler
# ===========================================================================
class ShowerSampler:
    def __init__(self, interp, Kmax=KMAX_DEFAULT):
        self.interp = interp
        self.Kmax = Kmax

    def sample_params(self, pid, E, rng):
        pm = self.interp.p_m(pid, E)
        # only sample m that actually have a fitted (mean,cov)
        avail = [m for m in range(1, self.Kmax + 1)
                 if self.interp.mean_cov(pid, E, m) is not None]
        p = np.array([pm[m - 1] for m in avail]); p = p / p.sum()
        m = int(rng.choice(avail, p=p))
        mean, cov = self.interp.mean_cov(pid, E, m)
        z = rng.multivariate_normal(mean, cov)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        z = np.clip(z, mean - 2.5 * sd, mean + 2.5 * sd)   # truncate fat Gaussian tails
        w, alpha, beta = _from_z(z, m)
        return m, w, alpha, beta

    def sample_profile(self, pid, E, rng, x=None, N=None, yield_sigma=None):
        if x is None:
            x = self.interp.z_centers(pid)
        m, w, alpha, beta = self.sample_params(pid, E, rng)
        if N is None:
            N = self.interp.yield_mean(pid, E)
            s = self.interp.yield_logsigma(pid, E) if yield_sigma is None else yield_sigma
            if s > 0:
                N *= np.exp(rng.normal(0, s))
        prof = N * np.sum([w[i] * _kernel(x, alpha[i], beta[i]) for i in range(m)], axis=0)
        return prof, dict(m=m, w=w, alpha=alpha, beta=beta, N=N)


# ===========================================================================
#  persistence : build once, save, load & sample later
# ===========================================================================
def save_model(interp, path):
    """Pickle a built ShowerParamInterpolator so you never refit."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(interp, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved model -> {path}")


def load_model(path):
    """Load a previously saved ShowerParamInterpolator."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_dists(dists, path):
    """Save the raw per-energy fit summaries (incl. the empirical (w,a,b) tuples)
    for diagnostics: decomposition, sampling-fidelity, and leave-one-out."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(dists, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved fit summaries -> {path}")


# ===========================================================================
#  Validation : are the sampled profiles physically correct?
# ===========================================================================
def validate_against_g4(library, sampler, pid, E, rng, n_sample=2000, out=None):
    """Overlay sampled mean+/-sigma vs the G4 truth ensemble at a grid energy,
    and compare total-yield and depth-of-max distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = library[pid][E]; x = d["z_centers"]; g4 = d["profiles"]
    samp = np.array([sampler.sample_profile(pid, E, rng, x=x)[0] for _ in range(n_sample)])

    def dmax(P): return x[np.argmax(P, axis=1)]
    g4_N, s_N = g4.sum(1), samp.sum(1)
    g4_dm, s_dm = dmax(g4), dmax(samp)

    print(f"\n=== validation  {PID_TO_NAME.get(pid,pid)}  E={E:.0f} GeV ===")
    print(f"  mean total yield   G4={g4_N.mean():.3g}   sampled={s_N.mean():.3g}")
    print(f"  yield sigma/mu     G4={g4_N.std()/g4_N.mean():.3f}   "
          f"sampled={s_N.std()/s_N.mean():.3f}")
    print(f"  depth-of-max [cm]  G4={g4_dm.mean():.0f}+/-{g4_dm.std():.0f}   "
          f"sampled={s_dm.mean():.0f}+/-{s_dm.std():.0f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(x, g4.mean(0), "k", lw=2, label="G4 mean")
    ax[0].fill_between(x, g4.mean(0) - g4.std(0), g4.mean(0) + g4.std(0),
                       color="k", alpha=0.15, label="G4 ±1σ")
    ax[0].plot(x, samp.mean(0), "r", lw=2, label="sampled mean")
    ax[0].fill_between(x, samp.mean(0) - samp.std(0), samp.mean(0) + samp.std(0),
                       color="r", alpha=0.15, label="sampled ±1σ")
    ax[0].set_xlabel("depth [cm]"); ax[0].set_ylabel("photons/bin"); ax[0].legend(fontsize=8)
    ax[0].set_title("mean ± σ profile")
    ax[1].hist(g4_N, 40, alpha=0.5, label="G4", color="k", density=True)
    ax[1].hist(s_N, 40, alpha=0.5, label="sampled", color="r", density=True)
    ax[1].set_xlabel("total yield"); ax[1].legend(fontsize=8); ax[1].set_title("yield dist")
    ax[2].hist(g4_dm, 30, alpha=0.5, label="G4", color="k", density=True)
    ax[2].hist(s_dm, 30, alpha=0.5, label="sampled", color="r", density=True)
    ax[2].set_xlabel("depth of max [cm]"); ax[2].legend(fontsize=8); ax[2].set_title("shower max")
    fig.suptitle(f"Sampled vs G4 — {PID_TO_NAME.get(pid,pid)} at {E:.0f} GeV")
    fig.tight_layout()
    if out is None:
        out = f"validate_{PID_TO_NAME.get(pid,pid)}_E{int(E)}GeV.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


# ===========================================================================
#  synthetic self-test : fabricate a library from a KNOWN model, then check
#  the pipeline recovers it and samples consistently.
# ===========================================================================
def _selftest():
    rng = np.random.default_rng(0)
    x = np.arange(2.5, 3000, 5.0)
    energies = [100.0, 1000.0, 10000.0, 100000.0]
    pid = 211
    # planted energy-dependent p(m): more multimodality at higher E
    def true_pm(E):
        t = (np.log10(E) - 2) / 3.0
        return np.array([0.8 - 0.5 * t, 0.15 + 0.35 * t, 0.05 + 0.15 * t])
    library = {pid: {}}
    for E in energies:
        pm = true_pm(E); pm = pm / pm.sum()
        Ntot = 3.0e4 * (E / 100.0)
        profs = []
        for _ in range(120):
            m = rng.choice([1, 2, 3], p=pm)
            centers = np.sort(rng.uniform(150, 700, m))
            w = rng.dirichlet(np.ones(m) * 3)
            prof = np.zeros_like(x)
            for c, wi in zip(centers, w):
                a = rng.uniform(4, 12); b = (a - 1) / c
                prof += wi * _kernel(x, a, b)
            prof = rng.poisson(np.maximum(Ntot * prof, 0)).astype(float)   # Poisson counts
            profs.append(prof)
        library[pid][E] = dict(profiles=np.array(profs),
                               N_total=np.array(profs).sum(1),
                               z_centers=x, z_edges=np.append(x - 2.5, x[-1] + 2.5))

    print("[selftest] fitting + aggregating ...")
    dists = build_distributions(library, verbose=True)
    interp = ShowerParamInterpolator(dists)
    sampler = ShowerSampler(interp)

    print("\n[selftest] recovered vs planted p(m):")
    ok = True
    for E in energies:
        rec = interp.p_m(pid, E); tru = true_pm(E); tru = tru / tru.sum()
        print(f"   E={E:7.0f}  planted={np.round(tru,2)}  recovered={np.round(rec,2)}")
        if abs(rec[0] - tru[0]) > 0.3:
            ok = False
    # interpolate at an OFF-grid energy and sample
    Eq = 3000.0
    prof, info = sampler.sample_profile(pid, Eq, rng)
    integ = _trapz(prof, x)
    print(f"\n[selftest] off-grid sample at E={Eq}: m={info['m']} "
          f"N={info['N']:.3g} integral/N={integ/info['N']:.2f} (~1 expected)")
    # consistency: sampled mean yield ~ interpolated N
    Ns = [sampler.sample_profile(pid, 1000.0, rng)[1]["N"] for _ in range(200)]
    print(f"[selftest] yield check at 1 TeV: sampled<N>={np.mean(Ns):.3g} "
          f"interp={interp.yield_mean(pid,1000.0):.3g}")
    print("\n[selftest] RESULT:", "PASS" if ok else "CHECK (p(m) recovery loose)")
    return ok


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--g4-dir", default=None)
    ap.add_argument("--species", default="pip")
    ap.add_argument("--energy", type=float, default=1000.0)
    ap.add_argument("--kmax", type=int, default=KMAX_DEFAULT)
    ap.add_argument("--save", default=None, help="pickle the built model to this path")
    ap.add_argument("--load", default=None, help="load a prebuilt model instead of refitting")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel fit workers (-1 = all cores)")
    ap.add_argument("--no-validate", action="store_true", help="skip the validation plot")
    args = ap.parse_args()

    if args.selftest or (args.g4_dir is None and args.load is None):
        _selftest(); return

    pid = NAME_TO_PID[args.species]
    lib = None

    if args.load:                         # load prebuilt model, skip all fitting
        interp = load_model(args.load)
        print(f"loaded model from {args.load}")
    else:                                 # build from G4 files (slow; save it!)
        lib = load_g4_library(args.g4_dir)
        dists = build_distributions(lib, Kmax=args.kmax, n_jobs=args.n_jobs)
        interp = ShowerParamInterpolator(dists, Kmax=args.kmax)
        if args.save:
            save_model(interp, args.save)
            save_dists(dists, args.save.replace(".pkl", "_dists.pkl"))

    sampler = ShowerSampler(interp, Kmax=args.kmax)

    if args.no_validate or args.g4_dir is None:
        prof, info = sampler.sample_profile(pid, args.energy, np.random.default_rng(1))
        print(f"[sample] {args.species} @ {args.energy:.0f} GeV -> "
              f"m={info['m']} N={info['N']:.3g} integral={_trapz(prof, sampler.interp.z_centers(pid)):.3g}")
    else:
        if lib is None:
            lib = load_g4_library(args.g4_dir)
        validate_against_g4(lib, sampler, pid, args.energy, np.random.default_rng(1),
                            out=os.path.join(args.g4_dir,
                                             f"validate_{args.species}_E{int(args.energy)}GeV.png"))


if __name__ == "__main__":
    main()
