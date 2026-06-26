#pragma once
#include "G4UserEventAction.hh"
#include <vector>

class RunAction;
class G4Event;

class EventAction : public G4UserEventAction {
public:
    // Longitudinal histogram: 600 bins x 5 cm = 30 m total depth
    static constexpr int    N_BINS      = 600;
    static constexpr double BIN_SIZE_CM = 5.0;

    explicit EventAction(RunAction* runAction);
    ~EventAction() = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*)   override;

    // Called by SteppingAction at each Cherenkov-producing step
    void AddCherenkov(double depth_cm, double dN);

private:
    RunAction*          fRunAction;
    std::vector<double> fHistogram;  // photon count per bin this event
    double              fNTotal;     // total photon count this event
};
