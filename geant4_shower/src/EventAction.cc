#include "EventAction.hh"
#include "RunAction.hh"

EventAction::EventAction(RunAction* runAction)
    : fRunAction(runAction),
      fHistogram(N_BINS, 0.0),
      fNTotal(0.0),
      fSubCounts{} {}

void EventAction::BeginOfEventAction(const G4Event*) {
    std::fill(fHistogram.begin(), fHistogram.end(), 0.0);
    fNTotal = 0.0;
    fSubCounts.fill(0);
}

void EventAction::AddCherenkov(double depth_cm, double dN) {
    int bin = static_cast<int>(depth_cm / BIN_SIZE_CM);
    if (bin >= 0 && bin < N_BINS)
        fHistogram[bin] += dN;
    fNTotal += dN;
}

void EventAction::AddInteraction(double ke_MeV) {
    // Count this interaction toward every threshold it clears. Thresholds are
    // fractions of the primary energy (from RunAction); G4 energies are in MeV.
    const double eprim_MeV = fRunAction->GetPrimaryEnergyGeV() * 1000.0;
    for (int i = 0; i < N_THRESH; ++i)
        if (ke_MeV > THRESH_FRAC[i] * eprim_MeV)
            ++fSubCounts[i];
}

void EventAction::EndOfEventAction(const G4Event*) {
    fRunAction->AddEvent(fHistogram, fNTotal, fSubCounts);
}
