#pragma once
#include "G4UserEventAction.hh"
#include <vector>
#include <array>

class RunAction;
class G4Event;

class EventAction : public G4UserEventAction {
public:
    // Longitudinal histogram: 600 bins x 5 cm = 30 m total depth
    static constexpr int    N_BINS      = 600;
    static constexpr double BIN_SIZE_CM = 5.0;

    // Sub-cascade counting: an inelastic hadronic interaction is counted toward
    // every threshold it clears; thresholds are fractions of the primary energy.
    static constexpr int    N_THRESH              = 5;
    static constexpr double THRESH_FRAC[N_THRESH] = {0.01, 0.02, 0.05, 0.10, 0.20};

    explicit EventAction(RunAction* runAction);
    ~EventAction() = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*)   override;

    // Called by SteppingAction at each Cherenkov-producing step
    void AddCherenkov(double depth_cm, double dN);

    // Called by SteppingAction at each inelastic hadronic interaction (any
    // particle); ke_MeV is the kinetic energy going into the interaction.
    void AddInteraction(double ke_MeV);

private:
    RunAction*                fRunAction;
    std::vector<double>       fHistogram;   // photon count per bin this event
    double                    fNTotal;      // total photon count this event
    std::array<int, N_THRESH> fSubCounts;   // sub-cascade counts per threshold
};
