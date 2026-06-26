#include "PrimaryGeneratorAction.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Event.hh"
#include "G4Exception.hh"

#include <string>

PrimaryGeneratorAction::PrimaryGeneratorAction(int pid, double energyGeV)
    : fPID(pid), fEnergyGeV(energyGeV)
{
    fGun = std::make_unique<G4ParticleGun>(1);

    // Start 10 cm upstream of the ice entry face (ice entry at z = -15000 mm)
    fGun->SetParticlePosition(G4ThreeVector(0, 0, -15100*mm));
    fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
    G4ParticleDefinition* particle =
        G4ParticleTable::GetParticleTable()->FindParticle(fPID);

    if (!particle) {
        G4String msg = "Unknown PDG ID: " + std::to_string(fPID)
                     + "\nSupported IDs: 211(pi+) -211(pi-) 111(pi0) "
                       "321(K+) -321(K-) 310(KS) 130(KL) 2212(p) 2112(n)";
        G4Exception("PrimaryGeneratorAction::GeneratePrimaries",
                    "UNKNOWN_PDG", FatalException, msg);
    }

    fGun->SetParticleDefinition(particle);
    // SetParticleEnergy sets KINETIC energy.
    // For E >> m (TeV scale) this is effectively the total energy.
    fGun->SetParticleEnergy(fEnergyGeV * GeV);
    fGun->GeneratePrimaryVertex(event);
}
