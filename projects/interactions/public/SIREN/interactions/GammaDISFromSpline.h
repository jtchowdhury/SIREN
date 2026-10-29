#pragma once
#ifndef SIREN_GammaDISFromSpline_H
#define SIREN_GammaDISFromSpline_H

// ---------------------------------------------------------------------------
//  GammaDISFromSpline
//
//  A child of DISFromSpline. The DIS kinematics (x, y, scattered lepton + Hadrons blob) 
//  are produced entirely by the base class. 
//  This class only ADDS the hadronic shower structure: it reads
//  E_had off the Hadrons secondary, picks a representative PYTHIA final state
//  conditioned on E_had, samples the per-particle gamma-mixture longitudinal
//  Cherenkov profile, and writes the components into the record's 
//  interaction_parameters as indexed scalars (the "gamma_subshowers" structure).
//
//  Two learned inputs:
//    1. shower model      -> shower_model_cpp.npz  (per-species gamma-mixture
//                            parameter curves + fluctuation covariances)
//    2. PYTHIA event lib  -> pythia_events_nue.npz (E_had + top-K secondary
//                            PDGs/energies, to pick a representative final state)
//  --------------------------------------------------------------------------

#include <set>
#include <map>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <cstdint>

#include "SIREN/dataclasses/InteractionSignature.h"

#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>

#include "SIREN/interactions/DISFromSpline.h"
#include "SIREN/dataclasses/Particle.h"

namespace siren { namespace dataclasses { struct InteractionRecord; } }
namespace siren { namespace dataclasses { class CrossSectionDistributionRecord; } }
namespace siren { namespace utilities { class SIREN_random; } }

namespace siren {
namespace interactions {

class GammaDISFromSpline : public DISFromSpline {
friend cereal::access;
public:
    // ---- one learned species' rebuilt parameter curves (host memory) -------
    struct SpeciesModel {
        int pid = 0;
        std::vector<double> logE;              // [L] training grid in log10(E/GeV)
        std::vector<double> z;                 // [Z] depth-bin centers [cm] (unused at sample time)
        std::vector<double> pm;                // [L*Kmax] p(m) over energy (row-major)
        std::vector<double> logN;              // [L] log mean yield
        std::vector<double> logNsig;           // [L] lognormal spread of the yield
        std::vector<int>    m_avail;           // which m have fitted params
        // per m: mean [L*(3m-1)] and cov [L*(3m-1)*(3m-1)], row-major
        std::map<int, std::vector<double>> mean;
        std::map<int, std::vector<double>> cov;
        double logE_lo = 0.0, logE_hi = 0.0;
    };

    // ---- a single gamma component handed back by the sampler ---------------
    struct Component { double alpha; double beta; double amplitude; };

private:
    std::string shower_model_path_;
    std::string pythia_events_path_;
    int    Kmax_ = 2;
    int    top_k_ = 10;
    int    n_bins_ = 12;
    bool   develop_lepton_ = true;             // CC nu_e: also shower the scattered e-

    // shower model, keyed by PDG
    std::map<int, SpeciesModel> species_;

    // PYTHIA event library (parallel arrays over events)
    std::vector<double>              pythia_Ehad_;      // [Nev]
    std::vector<std::vector<int>>    pythia_pids_;      // [Nev][<=20]
    std::vector<std::vector<double>> pythia_energies_;  // [Nev][<=20]
    // four-momenta for the Lorentz boost (require the updated pythia_events_nue.npz)
    std::vector<std::vector<std::array<double,4>>> pythia_momenta_; // [Nev][<=20][E,px,py,pz]
    std::vector<std::array<double,4>> pythia_phad_;     // [Nev] total hadronic 4-mom
    bool have_momenta_ = false;                          // true if the npz carried momenta
    std::vector<double>              bin_edges_;         // [n_bins+1] over E_had

    // our own signatures: lepton (mu/tau only) + kMaxSub_ GammaSubshower slots
    std::vector<dataclasses::InteractionSignature> gamma_signatures_;
    std::map<std::pair<siren::dataclasses::ParticleType, siren::dataclasses::ParticleType>,
             std::vector<dataclasses::InteractionSignature>> gamma_signatures_by_parents_;
    int kMaxSub_ = 0;                                    // padded max # of subshower slots

    bool loaded_ = false;

public:
    GammaDISFromSpline();

    // Short form -- mirrors the DISFromSpline(diff, total, primaries, targets,
    // units) ctor used by processes.py; interaction/target_mass/minimum_Q2 are
    // read from the spline table by the base. Appends the two model paths.
    GammaDISFromSpline(
        std::string differential_filename, std::string total_filename,
        std::vector<siren::dataclasses::ParticleType> primary_types,
        std::vector<siren::dataclasses::ParticleType> target_types,
        std::string shower_model_path, std::string pythia_events_path,
        std::string units = "cm");

    // Long form -- mirror the full DISFromSpline filename ctor and append the
    // two model paths. All DIS args are forwarded to the base ctor unchanged.
    GammaDISFromSpline(
        std::string differential_filename, std::string total_filename,
        int interaction, double target_mass, double minimum_Q2,
        std::vector<siren::dataclasses::ParticleType> primary_types,
        std::vector<siren::dataclasses::ParticleType> target_types,
        std::string shower_model_path, std::string pythia_events_path,
        std::string units = "cm");

    virtual ~GammaDISFromSpline() {}

    bool equal(CrossSection const & other) const override;

    // Reproduces the DIS kinematics (copied from DISFromSpline) but, instead of
    // emitting the scattered lepton + a single Hadrons blob, emits the hadronic
    // system (and, for CC nu_e, the electron) as GammaSubshower secondaries whose
    // 4-momentum slots carry [amplitude, alpha, beta, parent_pid]. For CC
    // nu_mu/nu_tau the lepton is kept as a normal track secondary (slot 0).
    void SampleFinalState(dataclasses::CrossSectionDistributionRecord &,
                          std::shared_ptr<siren::utilities::SIREN_random> random) const override;

    // Signature override: GammaSubshower slots (+ lepton slot for mu/tau). This
    // is what sets the secondary count; the base's [lepton, Hadrons] is replaced.
    std::vector<dataclasses::InteractionSignature> GetPossibleSignatures() const override;
    std::vector<dataclasses::InteractionSignature> GetPossibleSignaturesFromParents(
        siren::dataclasses::ParticleType primary_type,
        siren::dataclasses::ParticleType target_type) const override;

    std::string Name() const; // "GammaDISFromSpline"

    // ---- model access (public for unit tests / python) ---------------------
    void LoadShowerModel(std::string const & npz_path);   // numeric-only npz (see convert_model_for_cpp.py)
    void LoadPythiaEvents(std::string const & npz_path);
    bool HasSpecies(int pid) const { return species_.count(pid) > 0; }
    double YieldMean(int pid, double E_GeV) const;

    // Sample 1 particle's gamma mixture: returns its components (alpha, beta, amplitude = N * weight_i)
    std::vector<Component> SampleShower(
        int pid, double E_GeV,
        std::shared_ptr<siren::utilities::SIREN_random> random) const;

    // Pick a representative final state for this E_had -> list of (pid, E_GeV),
    // energy-conserving (top-K hadrons + a single pi0 remainder). 311 -> KS/KL.
    std::vector<std::pair<int,double>> PickFinalState(
        double E_had,
        std::shared_ptr<siren::utilities::SIREN_random> random) const;

    template<typename Archive>
    void save(Archive & archive, std::uint32_t const version) const {
        if(version == 0) {
            archive(::cereal::make_nvp("ShowerModelPath", shower_model_path_));
            archive(::cereal::make_nvp("PythiaEventsPath", pythia_events_path_));
            archive(::cereal::make_nvp("TopK", top_k_));
            archive(::cereal::make_nvp("DevelopLepton", develop_lepton_));
            archive(cereal::virtual_base_class<DISFromSpline>(this));
        } else {
            throw std::runtime_error("GammaDISFromSpline only supports version <= 0");
        }
    }

    template<typename Archive>
    void load(Archive & archive, std::uint32_t const version) {
        if(version == 0) {
            archive(::cereal::make_nvp("ShowerModelPath", shower_model_path_));
            archive(::cereal::make_nvp("PythiaEventsPath", pythia_events_path_));
            archive(::cereal::make_nvp("TopK", top_k_));
            archive(::cereal::make_nvp("DevelopLepton", develop_lepton_));
            archive(cereal::virtual_base_class<DISFromSpline>(this));
            if(!shower_model_path_.empty())  LoadShowerModel(shower_model_path_);
            if(!pythia_events_path_.empty()) LoadPythiaEvents(pythia_events_path_);
        } else {
            throw std::runtime_error("GammaDISFromSpline only supports version <= 0");
        }
    }

private:
    // numerics ported from hadronic_showers.py
    double interp1d_(std::vector<double> const & grid, double const * y, int ny_stride, int comp, double lq) const;
    double lq_clamped_(SpeciesModel const & s, double E) const;
    int    bin_index_(double E_had) const;
    static void from_z_(std::vector<double> const & z, int m,
                        std::vector<double> & w, std::vector<double> & alpha, std::vector<double> & beta);

    // build gamma_signatures_ from the primary/target types (CC assumed); called
    // once after the base ctor has set up primaries/targets.
    void InitializeGammaSignatures();

    // pick a pythia library event near E_had, Lorentz-boost its hadrons so their
    // total 4-momentum equals p4 (the DIS hadronic blob), and return the
    // energy-conserving (pid, E) breakdown (top-K + pi0 remainder). If the npz
    // has no momenta, falls back to uniform energy-scaling.
    // returns (parent_pid, boosted 4-mom [E,px,py,pz]) per final-state hadron
    // (top-K + pi0 remainder); the 4-momenta sum to p4 (conservation).
    std::vector<std::pair<int,std::array<double,4>>> BoostedBreakdown(
        std::array<double,4> const & p4,
        std::shared_ptr<siren::utilities::SIREN_random> random) const;
};

} // namespace interactions
} // namespace siren

CEREAL_CLASS_VERSION(siren::interactions::GammaDISFromSpline, 0);
CEREAL_REGISTER_TYPE(siren::interactions::GammaDISFromSpline);
CEREAL_REGISTER_POLYMORPHIC_RELATION(siren::interactions::CrossSection, siren::interactions::GammaDISFromSpline);
CEREAL_REGISTER_POLYMORPHIC_RELATION(siren::interactions::DISFromSpline, siren::interactions::GammaDISFromSpline);

#endif // SIREN_GammaDISFromSpline_H
