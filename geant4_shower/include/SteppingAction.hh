#pragma once
#include "G4UserSteppingAction.hh"

class EventAction;
class G4Step;

class SteppingAction : public G4UserSteppingAction {
public:
    explicit SteppingAction(EventAction* eventAction);
    ~SteppingAction() = default;
    void UserSteppingAction(const G4Step* step) override;

private:
    EventAction* fEventAction;
};
