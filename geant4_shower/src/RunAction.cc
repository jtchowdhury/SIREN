#include "RunAction.hh"
#include "EventAction.hh"

#include "hdf5.h"

#include <iostream>
#include <vector>
#include <string>

RunAction::RunAction(int pid, double energyGeV, const std::string& output)
    : fPID(pid), fEnergyGeV(energyGeV), fOutput(output) {}

void RunAction::BeginOfRunAction(const G4Run*) {
    fAllProfiles.clear();
    fAllNTotal.clear();
    fAllSubCounts.clear();
}

void RunAction::AddEvent(const std::vector<double>& histogram, double nTotal,
                         const std::array<int, EventAction::N_THRESH>& subCounts) {
    fAllProfiles.push_back(histogram);
    fAllNTotal.push_back(nTotal);
    fAllSubCounts.push_back(subCounts);
}

void RunAction::EndOfRunAction(const G4Run*) {
    // IsMaster() is true for single-threaded builds (always writes)
    if (IsMaster()) WriteHDF5();
}

void RunAction::WriteHDF5() const {
    const int nevents = static_cast<int>(fAllProfiles.size());
    const int nbins   = EventAction::N_BINS;

    // ── Flatten profiles to a row-major float array ───────────────────────
    std::vector<float> flat(nevents * nbins);
    for (int i = 0; i < nevents; ++i)
        for (int j = 0; j < nbins; ++j)
            flat[i * nbins + j] = static_cast<float>(fAllProfiles[i][j]);

    // ── Open file ─────────────────────────────────────────────────────────
    hid_t file = H5Fcreate(fOutput.c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "ERROR: cannot create HDF5 file: " << fOutput << "\n";
        return;
    }

    // ── profiles: (nevents, nbins) ────────────────────────────────────────
    {
        hsize_t dims[2] = {static_cast<hsize_t>(nevents),
                           static_cast<hsize_t>(nbins)};
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t dset  = H5Dcreate2(file, "profiles", H5T_NATIVE_FLOAT,
                                  space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 flat.data());
        H5Dclose(dset);
        H5Sclose(space);
    }

    // ── z_edges: (nbins+1,) in cm ─────────────────────────────────────────
    {
        std::vector<float> edges(nbins + 1);
        for (int i = 0; i <= nbins; ++i)
            edges[i] = i * static_cast<float>(EventAction::BIN_SIZE_CM);
        hsize_t dim = static_cast<hsize_t>(nbins + 1);
        hid_t space = H5Screate_simple(1, &dim, nullptr);
        hid_t dset  = H5Dcreate2(file, "z_edges", H5T_NATIVE_FLOAT,
                                  space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 edges.data());
        H5Dclose(dset);
        H5Sclose(space);
    }

    // ── N_total: (nevents,) ───────────────────────────────────────────────
    {
        std::vector<float> ntot(fAllNTotal.begin(), fAllNTotal.end());
        hsize_t dim = static_cast<hsize_t>(nevents);
        hid_t space = H5Screate_simple(1, &dim, nullptr);
        hid_t dset  = H5Dcreate2(file, "N_total", H5T_NATIVE_FLOAT,
                                  space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 ntot.data());
        H5Dclose(dset);
        H5Sclose(space);
    }

    // ── n_subcascades: (nevents, N_THRESH) int — count per fractional cut ──
    {
        const int nt = EventAction::N_THRESH;
        std::vector<int> sc(static_cast<size_t>(nevents) * nt);
        for (int i = 0; i < nevents; ++i)
            for (int j = 0; j < nt; ++j)
                sc[i * nt + j] = fAllSubCounts[i][j];
        hsize_t dims[2] = {static_cast<hsize_t>(nevents),
                           static_cast<hsize_t>(nt)};
        hid_t space = H5Screate_simple(2, dims, nullptr);
        hid_t dset  = H5Dcreate2(file, "n_subcascades", H5T_NATIVE_INT,
                                  space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sc.data());
        H5Dclose(dset);
        H5Sclose(space);
    }

    // ── Attributes ────────────────────────────────────────────────────────
    {
        hid_t scalar = H5Screate(H5S_SCALAR);

        hid_t attr = H5Acreate2(file, "pid", H5T_NATIVE_INT,
                                 scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &fPID);
        H5Aclose(attr);

        attr = H5Acreate2(file, "E_GeV", H5T_NATIVE_DOUBLE,
                          scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &fEnergyGeV);
        H5Aclose(attr);

        int n = nevents;
        attr = H5Acreate2(file, "n_events", H5T_NATIVE_INT,
                          scalar, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &n);
        H5Aclose(attr);

        H5Sclose(scalar);
    }

    // ── subcascade_thresholds: (N_THRESH,) the fractional cuts used ────────
    {
        hsize_t td = static_cast<hsize_t>(EventAction::N_THRESH);
        hid_t tspace = H5Screate_simple(1, &td, nullptr);
        hid_t tattr  = H5Acreate2(file, "subcascade_thresholds",
                                   H5T_NATIVE_DOUBLE, tspace,
                                   H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(tattr, H5T_NATIVE_DOUBLE, EventAction::THRESH_FRAC);
        H5Aclose(tattr);
        H5Sclose(tspace);
    }

    H5Fclose(file);
    std::cout << "Written: " << fOutput
              << "  (" << nevents << " events, "
              << nbins << " z-bins)\n";
}
