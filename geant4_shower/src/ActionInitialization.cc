#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(
    int pid, double energyGeV, int nevents, const std::string& output)
    : fPID(pid), fEnergyGeV(energyGeV), fNEvents(nevents), fOutput(output) {}

// Called only in multi-threaded mode for the master thread.
// Included for forward-compatibility; not used by G4RunManager (single-thread).
void ActionInitialization::BuildForMaster() const {
    SetUserAction(new RunAction(fPID, fEnergyGeV, fOutput));
}

void ActionInitialization::Build() const {
    auto* runAction   = new RunAction(fPID, fEnergyGeV, fOutput);
    auto* eventAction = new EventAction(runAction);

    SetUserAction(new PrimaryGeneratorAction(fPID, fEnergyGeV));
    SetUserAction(runAction);
    SetUserAction(eventAction);
    SetUserAction(new SteppingAction(eventAction));
}
