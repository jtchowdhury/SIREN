#include "DetectorConstruction.hh"
#include "ActionInitialization.hh"

#include "G4RunManager.hh"
#include "G4PhysListFactory.hh"
#include "G4VUserPhysicsList.hh"
#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>
#include <string>

// ── CLI argument struct ───────────────────────────────────────────────────────
struct Config {
    int         pid       = 211;          // PDG ID (default: pi+)
    double      energyGeV = 100.0;        // kinetic energy [GeV]
    int         nevents   = 1000;
    std::string output    = "output.h5";
};

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc - 1; ++i) {
        std::string a = argv[i];
        if      (a == "--pid")     cfg.pid       = std::stoi(argv[++i]);
        else if (a == "--energy")  cfg.energyGeV = std::stod(argv[++i]);
        else if (a == "--nevents") cfg.nevents    = std::stoi(argv[++i]);
        else if (a == "--output")  cfg.output     = argv[++i];
    }
    return cfg;
}

// ── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    std::cout << "=== G4ShowerSim ===\n"
              << "  PDG ID   : " << cfg.pid       << "\n"
              << "  Energy   : " << cfg.energyGeV << " GeV (kinetic)\n"
              << "  N events : " << cfg.nevents    << "\n"
              << "  Output   : " << cfg.output     << "\n\n";

    // Run manager (single-threaded — one job per species/energy on HPC)
    auto* runManager = new G4RunManager();

    // Detector: ice cylinder
    runManager->SetUserInitialization(new DetectorConstruction());

    // Physics: FTFP_BERT, 1 mm production cuts, no optical physics
    G4PhysListFactory factory;
    G4VUserPhysicsList* phys = factory.GetReferencePhysList("FTFP_BERT");
    phys->SetDefaultCutValue(1.0 * CLHEP::mm);
    runManager->SetUserInitialization(phys);

    // User actions: generator, event, stepping, run
    runManager->SetUserInitialization(
        new ActionInitialization(cfg.pid, cfg.energyGeV, cfg.nevents, cfg.output));

    runManager->Initialize();
    runManager->BeamOn(cfg.nevents);

    delete runManager;
    return 0;
}
