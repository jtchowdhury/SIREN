cmake_minimum_required(VERSION 3.16)
project(G4ShowerSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Geant4 ────────────────────────────────────────────────────────────────────
find_package(Geant4 REQUIRED)
include(${Geant4_USE_FILE})

# ── HDF5 (C component only) ───────────────────────────────────────────────────
find_package(HDF5 REQUIRED COMPONENTS C)

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    sim.cc
    src/DetectorConstruction.cc
    src/ActionInitialization.cc
    src/PrimaryGeneratorAction.cc
    src/EventAction.cc
    src/RunAction.cc
    src/SteppingAction.cc
)

add_executable(sim ${SOURCES})
target_include_directories(sim PRIVATE include ${HDF5_INCLUDE_DIRS})
target_link_libraries(sim PRIVATE ${Geant4_LIBRARIES} ${HDF5_C_LIBRARIES})
