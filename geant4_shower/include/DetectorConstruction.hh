#pragma once
#include "G4VUserDetectorConstruction.hh"

class G4VPhysicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    DetectorConstruction()  = default;
    ~DetectorConstruction() = default;
    G4VPhysicalVolume* Construct() override;
};
