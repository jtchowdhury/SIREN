#include "DetectorConstruction.hh"

#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4SystemOfUnits.hh"

G4VPhysicalVolume* DetectorConstruction::Construct() {
    G4NistManager* nist = G4NistManager::Instance();

    // ── World: vacuum box, large enough to contain the ice + gun ─────────────
    G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* worldBox = new G4Box("World", 20000*mm, 20000*mm, 20000*mm);
    auto* worldLog = new G4LogicalVolume(worldBox, vacuum, "World");
    auto* worldPhys = new G4PVPlacement(
        nullptr, G4ThreeVector(0,0,0), worldLog, "World", nullptr, false, 0, true);

    // ── Ice: H2O at ice density ───────────────────────────────────────────────
    // Composition identical to water; density 0.917 g/cm3 for bulk ice.
    G4Element* H = nist->FindOrBuildElement("H");
    G4Element* O = nist->FindOrBuildElement("O");
    auto* ice = new G4Material("ICE", 0.917*g/cm3, 2, kStateSolid);
    ice->AddElement(H, 2);
    ice->AddElement(O, 1);

    // ── Ice cylinder ─────────────────────────────────────────────────────────
    // Half-length 15 m, radius 5 m, centered at world origin.
    // Entry face at z = -15000 mm, exit face at z = +15000 mm.
    // 30 m is sufficient to contain hadronic showers up to ~100 TeV.
    auto* iceTubs = new G4Tubs("IceTubs", 0, 5000*mm, 15000*mm, 0, 360*deg);
    auto* iceLog  = new G4LogicalVolume(iceTubs, ice, "IceCylinder_log");
    new G4PVPlacement(
        nullptr, G4ThreeVector(0,0,0), iceLog, "IceCylinder", worldLog, false, 0, true);

    return worldPhys;
}
