#pragma once
#include "G4VUserActionInitialization.hh"
#include <string>

class ActionInitialization : public G4VUserActionInitialization {
public:
    ActionInitialization(int pid, double energyGeV, int nevents, const std::string& output);
    void BuildForMaster() const override;
    void Build() const override;

private:
    int         fPID;
    double      fEnergyGeV;
    int         fNEvents;
    std::string fOutput;
};
