#pragma once
#include "G4UserRunAction.hh"
#include "EventAction.hh"
#include <vector>
#include <string>
#include <array>

class G4Run;

class RunAction : public G4UserRunAction {
public:
    RunAction(int pid, double energyGeV, const std::string& output);
    ~RunAction() = default;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*)   override;

    // Called by EventAction at end of each event
    void AddEvent(const std::vector<double>& histogram, double nTotal,
                  const std::array<int, EventAction::N_THRESH>& subCounts);

    double GetPrimaryEnergyGeV() const { return fEnergyGeV; }

private:
    void WriteHDF5() const;

    int         fPID;
    double      fEnergyGeV;
    std::string fOutput;

    std::vector<std::vector<double>>                    fAllProfiles;
    std::vector<double>                                 fAllNTotal;
    std::vector<std::array<int, EventAction::N_THRESH>> fAllSubCounts;
};
