//  Model file: a NUMERIC-ONLY repacking of shower_model.npz produced by
//  resources/analysis/convert_model_for_cpp.py. We avoid the original file's
//  __meta__ JSON (a unicode numpy array) because cnpy reads numeric arrays, not
//  unicode/JSON. The repack adds integer index arrays "pids", "Kmax", and per
//  species "<pid>|m_avail"; every other array is copied through unchanged.
// ---------------------------------------------------------------------------

#include "SIREN/interactions/GammaDISFromSpline.h"

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "cnpy.h"   // vendored, header-only; see third_party/cnpy (MIT)

#include "SIREN/dataclasses/InteractionRecord.h"
#include "SIREN/dataclasses/Particle.h"
#include "SIREN/utilities/Random.h"

namespace siren {
namespace interactions {

namespace {
    constexpr double CLIP_SIGMA = 2.5;   // matches training (hadronic_showers._CLIP_SIGMA)
    constexpr double COV_RIDGE  = 1e-3;  // PSD floor (hadronic_showers._COV_RIDGE)
    constexpr int    HADRONS_PDG = -2000001006;
    constexpr int    ELECTRON_PDG = 11;

    // standard normal via Box-Muller from SIREN_random::Uniform (keeps us off any
    // particular Gaussian API; statistically identical)
    double draw_normal(std::shared_ptr<siren::utilities::SIREN_random> r) {
        double u1 = std::max(r->Uniform(0.0, 1.0), 1e-300);
        double u2 = r->Uniform(0.0, 1.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }

    int pdg_to_model_pid(int pid, std::shared_ptr<siren::utilities::SIREN_random> r) {
        // K0/K0bar -> KS(310)/KL(130) 50/50 (mirrors load_final_state_h5)
        if(std::abs(pid) == 311) return (r->Uniform(0.0,1.0) < 0.5) ? 310 : 130;
        return pid;
    }
} // namespace

// ===========================================================================
//  construction
// ===========================================================================
GammaDISFromSpline::GammaDISFromSpline() : DISFromSpline() {}

GammaDISFromSpline::GammaDISFromSpline(
    std::string differential_filename, std::string total_filename,
    std::vector<siren::dataclasses::ParticleType> primary_types,
    std::vector<siren::dataclasses::ParticleType> target_types,
    std::string shower_model_path, std::string pythia_events_path,
    std::string units)
    : DISFromSpline(differential_filename, total_filename,
                    primary_types, target_types, units),
      shower_model_path_(shower_model_path),
      pythia_events_path_(pythia_events_path) {
    LoadShowerModel(shower_model_path_);
    LoadPythiaEvents(pythia_events_path_);
}

GammaDISFromSpline::GammaDISFromSpline(
    std::string differential_filename, std::string total_filename,
    int interaction, double target_mass, double minimum_Q2,
    std::vector<siren::dataclasses::ParticleType> primary_types,
    std::vector<siren::dataclasses::ParticleType> target_types,
    std::string shower_model_path, std::string pythia_events_path,
    std::string units)
    : DISFromSpline(differential_filename, total_filename, interaction,
                    target_mass, minimum_Q2, primary_types, target_types, units),
      shower_model_path_(shower_model_path),
      pythia_events_path_(pythia_events_path) {
    LoadShowerModel(shower_model_path_);
    LoadPythiaEvents(pythia_events_path_);
}

std::string GammaDISFromSpline::Name() const { return "GammaDISFromSpline"; }

bool GammaDISFromSpline::equal(CrossSection const & other) const {
    const GammaDISFromSpline* x = dynamic_cast<const GammaDISFromSpline*>(&other);
    if(!x) return false;
    return shower_model_path_ == x->shower_model_path_ &&
           pythia_events_path_ == x->pythia_events_path_ &&
           DISFromSpline::equal(other);
}

// ===========================================================================
//  model loading (cnpy)
// ===========================================================================
void GammaDISFromSpline::LoadShowerModel(std::string const & npz_path) {
    cnpy::npz_t npz = cnpy::npz_load(npz_path);
    species_.clear();

    auto get_d = [&](std::string const & k) -> cnpy::NpyArray & {
        auto it = npz.find(k);
        if(it == npz.end()) throw std::runtime_error("shower model missing key: " + k);
        return it->second;
    };

    cnpy::NpyArray & pids_arr = get_d("pids");
    const int64_t* pids = pids_arr.data<int64_t>();
    Kmax_ = static_cast<int>(get_d("Kmax").data<int64_t>()[0]);
    size_t P = pids_arr.shape[0];

    for(size_t ip = 0; ip < P; ++ip) {
        int pid = static_cast<int>(pids[ip]);
        std::string p = std::to_string(pid);
        SpeciesModel s;
        s.pid = pid;

        auto & logE = get_d(p + "|logE");
        s.logE.assign(logE.data<double>(), logE.data<double>() + logE.shape[0]);
        s.logE_lo = s.logE.front();
        s.logE_hi = s.logE.back();

        auto & z = get_d(p + "|z");
        s.z.assign(z.data<double>(), z.data<double>() + z.shape[0]);

        auto & pm = get_d(p + "|pm");       // [L, Kmax]
        s.pm.assign(pm.data<double>(), pm.data<double>() + pm.num_vals);

        auto & lN = get_d(p + "|logN");
        s.logN.assign(lN.data<double>(), lN.data<double>() + lN.shape[0]);
        auto & lNs = get_d(p + "|logNsig");
        s.logNsig.assign(lNs.data<double>(), lNs.data<double>() + lNs.shape[0]);

        auto & ma = get_d(p + "|m_avail");
        const int64_t* mv = ma.data<int64_t>();
        for(size_t j = 0; j < ma.shape[0]; ++j) {
            int m = static_cast<int>(mv[j]);
            s.m_avail.push_back(m);
            auto & mean = get_d(p + "|m" + std::to_string(m) + "|mean"); // [L, 3m-1]
            auto & cov  = get_d(p + "|m" + std::to_string(m) + "|cov");  // [L, 3m-1, 3m-1]
            s.mean[m].assign(mean.data<double>(), mean.data<double>() + mean.num_vals);
            s.cov[m].assign(cov.data<double>(),  cov.data<double>()  + cov.num_vals);
        }
        species_[pid] = std::move(s);
    }
    loaded_ = true;
}

void GammaDISFromSpline::LoadPythiaEvents(std::string const & npz_path) {
    cnpy::npz_t npz = cnpy::npz_load(npz_path);
    auto get = [&](std::string const & k) -> cnpy::NpyArray & {
        auto it = npz.find(k);
        if(it == npz.end()) throw std::runtime_error("pythia store missing key: " + k);
        return it->second;
    };
    cnpy::NpyArray & EH = get("E_had");          // [Nev]
    cnpy::NpyArray & PD = get("top20_pids");     // [Nev, 20] int64
    cnpy::NpyArray & EN = get("top20_energies"); // [Nev, 20] double
    size_t Nev = EH.shape[0];
    size_t cols = PD.shape.size() > 1 ? PD.shape[1] : 20;

    const double*  eh = EH.data<double>();
    const int64_t* pd = PD.data<int64_t>();
    const double*  en = EN.data<double>();

    pythia_Ehad_.assign(eh, eh + Nev);
    pythia_pids_.resize(Nev);
    pythia_energies_.resize(Nev);
    for(size_t i = 0; i < Nev; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            pythia_pids_[i].push_back(static_cast<int>(pd[i*cols + j]));
            pythia_energies_[i].push_back(en[i*cols + j]);
        }
    }

    // quantile bin edges over E_had (mirrors hadronic_showers.event_bins)
    std::vector<double> sorted = pythia_Ehad_;
    std::sort(sorted.begin(), sorted.end());
    bin_edges_.resize(n_bins_ + 1);
    for(int b = 0; b <= n_bins_; ++b) {
        double q = double(b) / n_bins_;
        double pos = q * (sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(pos));
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = pos - lo;
        bin_edges_[b] = sorted[lo] * (1 - frac) + sorted[hi] * frac;
    }
}

// ===========================================================================
//  numerics ported from hadronic_showers.py
// ===========================================================================

// linear interpolation of y[:,comp] (stride ny_stride) over `grid` at lq (clamped)
double GammaDISFromSpline::interp1d_(std::vector<double> const & grid,
                                     double const * y, int ny_stride, int comp,
                                     double lq) const {
    size_t n = grid.size();
    if(n == 1) return y[comp];
    if(lq <= grid.front()) return y[0*ny_stride + comp];
    if(lq >= grid.back())  return y[(n-1)*ny_stride + comp];
    size_t hi = std::lower_bound(grid.begin(), grid.end(), lq) - grid.begin();
    size_t lo = hi - 1;
    double t = (lq - grid[lo]) / (grid[hi] - grid[lo]);
    return y[lo*ny_stride + comp] * (1 - t) + y[hi*ny_stride + comp] * t;
}

double GammaDISFromSpline::lq_clamped_(SpeciesModel const & s, double E) const {
    double lq = std::log10(E);
    return std::min(std::max(lq, s.logE_lo), s.logE_hi);
}

// (log centroid, log width) -> (alpha, beta); inverse of the training encoding
// alpha = (c/s)^2 ; beta = c/s^2   (hadronic_showers._comp_from_z)
void GammaDISFromSpline::from_z_(std::vector<double> const & z, int m,
                                 std::vector<double> & w,
                                 std::vector<double> & alpha,
                                 std::vector<double> & beta) {
    w.assign(m, 1.0); alpha.assign(m, 1.0); beta.assign(m, 1.0);
    if(m == 1) {
        double c = std::exp(z[0]), s = std::exp(z[1]);
        alpha[0] = (c/s)*(c/s);
        beta[0]  = c/(s*s);
        return;
    }
    // z = [alr(m-1), lc(m), ls(m)]
    double denom = 1.0; // exp(0) for the pinned last component
    std::vector<double> e(m);
    for(int i = 0; i < m - 1; ++i) { e[i] = std::exp(z[i]); denom += e[i]; }
    e[m-1] = 1.0;
    for(int i = 0; i < m; ++i) w[i] = e[i] / denom;
    for(int i = 0; i < m; ++i) {
        double c = std::exp(z[(m-1) + i]);
        double s = std::exp(z[(m-1) + m + i]);
        alpha[i] = (c/s)*(c/s);
        beta[i]  = c/(s*s);
    }
}

double GammaDISFromSpline::YieldMean(int pid, double E_GeV) const {
    auto it = species_.find(pid);
    if(it == species_.end()) return 0.0;
    SpeciesModel const & s = it->second;
    double lq = lq_clamped_(s, E_GeV);
    return std::exp(interp1d_(s.logE, s.logN.data(), 1, 0, lq));
}

// ===========================================================================
//  one-particle sampler -- faithful port of HadronicShowerModel.sample
// ===========================================================================
std::vector<GammaDISFromSpline::Component> GammaDISFromSpline::SampleShower(
    int pid, double E_GeV,
    std::shared_ptr<siren::utilities::SIREN_random> random) const {

    std::vector<Component> out;
    auto it = species_.find(pid);
    if(it == species_.end()) return out;      // unknown species: no components
    SpeciesModel const & s = it->second;
    double lq = lq_clamped_(s, E_GeV);

    // ---- pick m among the fitted ones, weighted by p(m) --------------------
    std::vector<double> p; p.reserve(s.m_avail.size());
    double psum = 0.0;
    for(int m : s.m_avail) {
        double pm = interp1d_(s.logE, s.pm.data(), Kmax_, m - 1, lq);
        pm = std::max(pm, 0.0);
        p.push_back(pm); psum += pm;
    }
    if(psum <= 0.0) { for(auto & v : p) v = 1.0 / p.size(); }
    else            { for(auto & v : p) v /= psum; }
    double u = random->Uniform(0.0, 1.0);
    int m = s.m_avail.back(), idx = 0; double acc = 0.0;
    for(size_t i = 0; i < p.size(); ++i) { acc += p[i]; if(u <= acc) { m = s.m_avail[i]; idx = i; break; } }
    (void)idx;

    int dim = 3 * m - 1;

    // ---- mean + cov at lq (interpolated per element) -----------------------
    std::vector<double> mean(dim), cov(dim * dim);
    auto const & meanbuf = s.mean.at(m);   // [L, dim]
    auto const & covbuf  = s.cov.at(m);    // [L, dim*dim]
    for(int c = 0; c < dim; ++c)
        mean[c] = interp1d_(s.logE, meanbuf.data(), dim, c, lq);
    for(int c = 0; c < dim*dim; ++c)
        cov[c] = interp1d_(s.logE, covbuf.data(), dim*dim, c, lq);

    // symmetrize + diagonal ridge (PSD safeguard; Python clips eigenvalues to
    // COV_RIDGE, here we add it to the diagonal before Cholesky)
    for(int a = 0; a < dim; ++a)
        for(int b = a+1; b < dim; ++b) {
            double v = 0.5 * (cov[a*dim+b] + cov[b*dim+a]);
            cov[a*dim+b] = cov[b*dim+a] = v;
        }
    for(int a = 0; a < dim; ++a) cov[a*dim+a] += COV_RIDGE;

    // ---- Cholesky cov = L L^T ---------------------------------------------
    std::vector<double> L(dim * dim, 0.0);
    for(int a = 0; a < dim; ++a) {
        for(int b = 0; b <= a; ++b) {
            double sum = cov[a*dim + b];
            for(int k = 0; k < b; ++k) sum -= L[a*dim+k] * L[b*dim+k];
            if(a == b) L[a*dim+b] = std::sqrt(std::max(sum, 1e-12));
            else       L[a*dim+b] = sum / L[b*dim+b];
        }
    }

    // ---- z = mean + L g,  g ~ N(0,1); then clip to +-CLIP_SIGMA ------------
    std::vector<double> g(dim), z(dim);
    for(int a = 0; a < dim; ++a) g[a] = draw_normal(random);
    for(int a = 0; a < dim; ++a) {
        double v = mean[a];
        for(int b = 0; b <= a; ++b) v += L[a*dim+b] * g[b];
        double sd = std::sqrt(std::max(cov[a*dim+a], 0.0));
        z[a] = std::min(std::max(v, mean[a] - CLIP_SIGMA*sd), mean[a] + CLIP_SIGMA*sd);
    }

    std::vector<double> w, alpha, beta;
    from_z_(z, m, w, alpha, beta);

    // ---- yield N with lognormal fluctuation, scaled ~E out of range --------
    double N = std::exp(interp1d_(s.logE, s.logN.data(), 1, 0, lq));
    double sig = std::max(interp1d_(s.logE, s.logNsig.data(), 1, 0, lq), 0.0);
    if(sig > 0.0) N *= std::exp(sig * draw_normal(random));
    N *= E_GeV / std::pow(10.0, lq);   // == 1 in-range; yield ~ E out of range

    out.reserve(m);
    for(int i = 0; i < m; ++i)
        out.push_back({alpha[i], beta[i], N * w[i]});
    return out;
}

// ===========================================================================
//  representative final state for an E_had (energy-conserving top-K + pi0)
// ===========================================================================
int GammaDISFromSpline::bin_index_(double E_had) const {
    if(bin_edges_.empty()) return 0;
    int b = 0;
    while(b < n_bins_ && E_had >= bin_edges_[b+1]) ++b;
    return std::min(std::max(b, 0), n_bins_ - 1);
}

std::vector<std::pair<int,double>> GammaDISFromSpline::PickFinalState(
    double E_had, std::shared_ptr<siren::utilities::SIREN_random> random) const {

    std::vector<std::pair<int,double>> fs;
    if(pythia_Ehad_.empty()) return fs;

    // nearest non-empty bin, then a uniform event in it
    int b = bin_index_(E_had);
    std::vector<size_t> pool;
    for(int radius = 0; radius < n_bins_ && pool.empty(); ++radius) {
        for(int bb : {b - radius, b + radius}) {
            if(bb < 0 || bb >= n_bins_) continue;
            double lo = bin_edges_[bb], hi = bin_edges_[bb+1];
            for(size_t i = 0; i < pythia_Ehad_.size(); ++i)
                if(pythia_Ehad_[i] >= lo && (pythia_Ehad_[i] < hi || bb == n_bins_-1))
                    pool.push_back(i);
            if(radius == 0) break; // first pass: only the home bin
        }
    }
    if(pool.empty()) { for(size_t i=0;i<pythia_Ehad_.size();++i) pool.push_back(i); }
    size_t ievt = pool[std::min<size_t>(pool.size()-1,
                      static_cast<size_t>(random->Uniform(0.0,1.0) * pool.size()))];

    double scale = (pythia_Ehad_[ievt] > 0.0) ? (E_had / pythia_Ehad_[ievt]) : 1.0;

    // rank the event's secondaries by energy, keep top_k known hadrons, fold the
    // rest into one pi0 (energy-conserving). (Matches top_k_final_state; the
    // gamma/e+-/pbar/nbar routing refinement is still pending -- see handoff.)
    std::vector<std::pair<double,int>> ranked;
    for(size_t j = 0; j < pythia_pids_[ievt].size(); ++j) {
        int pid = pythia_pids_[ievt][j];
        double E = pythia_energies_[ievt][j] * scale;
        if(pid == 0 || !(E > 0.0)) continue;
        ranked.push_back({E, pid});
    }
    std::sort(ranked.begin(), ranked.end(), std::greater<>());

    double remainder = 0.0;
    int kept = 0;
    for(auto & pe : ranked) {
        int pid = pdg_to_model_pid(pe.second, random);
        if(kept < top_k_ && HasSpecies(pid)) { fs.push_back({pid, pe.first}); ++kept; }
        else remainder += pe.first;
    }
    if(remainder > 0.0 && HasSpecies(111)) fs.push_back({111, remainder}); // pi0
    return fs;
}

// ===========================================================================
//  the override: base DIS kinematics, then add gamma subshowers (indexed params)
// ===========================================================================
void GammaDISFromSpline::SampleFinalState(
    dataclasses::CrossSectionDistributionRecord & record,
    std::shared_ptr<siren::utilities::SIREN_random> random) const {

    // 1) let the base produce the scattered lepton + Hadrons blob (sets x,y,E_had)
    DISFromSpline::SampleFinalState(record, random);

    // 2) read E_had off the Hadrons secondary; capture the scattered e- (CC nu_e)
    double E_had = 0.0, E_lepton = 0.0;
    bool have_had = false, have_e = false;
    auto & secondaries = record.GetSecondaryParticleRecords();
    for(auto & sec : secondaries) {
        int pdg = static_cast<int>(sec.GetType());
        if(pdg == HADRONS_PDG) { E_had = sec.GetEnergy(); have_had = true; }
        else if(std::abs(pdg) == ELECTRON_PDG) { E_lepton = sec.GetEnergy(); have_e = true; }
    }
    if(!have_had || E_had <= 0.0) return;   // nothing to develop

    // 3) representative final state, plus the scattered electron if present
    auto particles = PickFinalState(E_had, random);
    if(develop_lepton_ && have_e && HasSpecies(ELECTRON_PDG))
        particles.push_back({ELECTRON_PDG, E_lepton});

    // 4) sample each particle's mixture; write indexed interaction_parameters.
    //    Layout (all doubles):
    //      subshower_count
    //      subshower_<i>_alpha / _beta / _amplitude
    //      subshower_<i>_parent_pid   (PDG of the parent particle)
    //      subshower_<i>_parent_inst  (k-th particle of that pid in this event)
    //    Components with the same (parent_pid, parent_inst) are one m>=2 particle.
    auto & params = record.GetInteractionParameters();
    std::map<int,int> inst;   // pid -> running instance count
    int idx = 0;
    for(auto & pe : particles) {
        int pid = pe.first; double E = pe.second;
        int k = ++inst[pid];
        auto comps = SampleShower(pid, E, random);
        for(auto & c : comps) {
            std::string pre = "subshower_" + std::to_string(idx);
            params[pre + "_alpha"]       = c.alpha;
            params[pre + "_beta"]        = c.beta;
            params[pre + "_amplitude"]   = c.amplitude;
            params[pre + "_parent_pid"]  = static_cast<double>(pid);
            params[pre + "_parent_inst"] = static_cast<double>(k);
            ++idx;
        }
    }
    params["subshower_count"] = static_cast<double>(idx);
}

} // namespace interactions
} // namespace siren
