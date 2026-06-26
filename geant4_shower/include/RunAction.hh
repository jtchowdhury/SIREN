#pragma once
#include "G4UserRunAction.hh"
#include <vector>
#include <string>

class G4Run;

class RunAction : public G4UserRunAction {
public:
    RunAction(int pid, double energyGeV, const std::string& output);
    ~RunAction() = default;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*)   override;

    // Called by EventAction at end of each event
    void AddEvent(const std::vector<double>& histogram, double nTotal);

private:
    void WriteHDF5() const;

    int         fPID;
    double      fEnergyGeV;
    std::string fOutput;

    std::vector<std::vector<double>> fAllProfiles;
    std::vector<double>              fAllNTotal;
};
