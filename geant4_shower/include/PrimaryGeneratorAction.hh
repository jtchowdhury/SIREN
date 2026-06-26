#pragma once
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include <memory>

class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    PrimaryGeneratorAction(int pid, double energyGeV);
    ~PrimaryGeneratorAction() = default;
    void GeneratePrimaries(G4Event* event) override;

private:
    std::unique_ptr<G4ParticleGun> fGun;
    int    fPID;
    double fEnergyGeV;
};
