#include "EventAction.hh"
#include "RunAction.hh"

EventAction::EventAction(RunAction* runAction)
    : fRunAction(runAction),
      fHistogram(N_BINS, 0.0),
      fNTotal(0.0) {}

void EventAction::BeginOfEventAction(const G4Event*) {
    std::fill(fHistogram.begin(), fHistogram.end(), 0.0);
    fNTotal = 0.0;
}

void EventAction::AddCherenkov(double depth_cm, double dN) {
    int bin = static_cast<int>(depth_cm / BIN_SIZE_CM);
    if (bin >= 0 && bin < N_BINS)
        fHistogram[bin] += dN;
    fNTotal += dN;
}

void EventAction::EndOfEventAction(const G4Event*) {
    fRunAction->AddEvent(fHistogram, fNTotal);
}
