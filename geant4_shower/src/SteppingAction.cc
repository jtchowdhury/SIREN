#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4ThreeVector.hh"
#include "G4VPhysicalVolume.hh"
#include "G4TouchableHandle.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"

#include <cmath>
#include <string>

SteppingAction::SteppingAction(EventAction* eventAction)
    : fEventAction(eventAction) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {

    const G4Track* track = step->GetTrack();

    // 1. Only consider steps inside the ice volume.
    //    GetVolume() returns nullptr outside the world, so guard against that.
    const G4VPhysicalVolume* vol =
        step->GetPreStepPoint()->GetTouchableHandle()->GetVolume();
    if (!vol || vol->GetName() != "IceCylinder") return;

    // 2. Sub-cascade counting.
    //    Count every inelastic hadronic interaction — for ANY particle,
    //    including neutrons/K0 (so this must happen before the charge cut).
    //    The pre-step kinetic energy is the energy entering the interaction;
    //    EventAction applies the fractional-of-primary thresholds.
    {
        const G4VProcess* proc =
            step->GetPostStepPoint()->GetProcessDefinedStep();
        if (proc &&
            proc->GetProcessName().find("Inelastic") != std::string::npos) {
            fEventAction->AddInteraction(
                step->GetPreStepPoint()->GetKineticEnergy());
        }
    }

    // 3. Cherenkov radiation is only produced by charged particles.
    if (track->GetDefinition()->GetPDGCharge() == 0.0) return;

    // 4. Frank-Tamm Cherenkov yield ──────────────────────────────────────────
    //
    //   dN/dx = 2π α sin²θ_C (1/λ_min - 1/λ_max)
    //
    //   sin²θ_C = 1 - 1/(n² β²)
    //
    //   n_ice = 1.33  (average over 300–600 nm; wavelength dependence adds
    //                  ~10–15% correction and can be included later)
    //   λ range: 300–600 nm matches IceCube DOM sensitivity
    //   All lengths in Geant4 internal units (mm)

    const double n_ice = 1.33;
    const double beta  = step->GetPreStepPoint()->GetBeta();

    const double cos2_theta_c = 1.0 / (n_ice * n_ice * beta * beta);
    if (cos2_theta_c >= 1.0) return;  // particle below Cherenkov threshold

    const double sin2_theta_c = 1.0 - cos2_theta_c;

    constexpr double alpha      = 1.0 / 137.036;
    constexpr double lambda_min = 300.0e-6;  // mm (300 nm)
    constexpr double lambda_max = 600.0e-6;  // mm (600 nm)

    const double step_len = step->GetStepLength();  // mm

    const double dN = 2.0 * M_PI * alpha * sin2_theta_c
                      * (1.0 / lambda_min - 1.0 / lambda_max)
                      * step_len;

    // 4. Convert z-position to depth in ice (cm).
    //    Ice cylinder: z from -15000 mm to +15000 mm.
    //    depth = z + 15000 mm, converted to cm.
    const double z_mm     = step->GetPreStepPoint()->GetPosition().z();
    const double depth_cm = (z_mm + 15000.0) / 10.0;

    fEventAction->AddCherenkov(depth_cm, dN);
}
