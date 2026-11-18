#!/bin/bash
#SBATCH -J gammadis-validation
#SBATCH -p shared
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o gammadis_validation_%j.log
#
# Samples and plots for the GammaDISFromSpline validation, in order.
#
#   interactive:  bash run_gamma_dis_validation.sh 2>&1 | tee validation.log
#   batch:        sbatch run_gamma_dis_validation.sh
#
# Steps can be switched off, e.g. to redo only the plots:
#   SAMPLES=0 bash run_gamma_dis_validation.sh
#
#   SAMPLES=0      skip all sample generation
#   SPECTRUM=0     skip the spread-energy samples (slow: Pythia is set up again for
#                  every event); the kinematics range plots and the reweighting plots
#                  need them
#   LIBRARIES=0    skip the Pythia reference libraries for the CC/NC plot
#   PLOTS=0        skip the plots
#
# A failed step is reported and the script continues with the next one.

N_MONO=${N_MONO:-10000}          # fixed-energy kinematics samples (fast)
N_SPECTRUM=${N_SPECTRUM:-3000}   # spread-energy samples (slow)
N_CCNC=${N_CCNC:-500}            # CC/NC events per energy, 100 GeV - 1 PeV
N_LIB=${N_LIB:-3000}             # Pythia reference events per energy

if ! command -v conda > /dev/null; then module load python 2> /dev/null; fi
eval "$(conda shell.bash hook)"
conda activate siren-dev
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export SIREN_ROOT=${SIREN_ROOT:-$HOME/SIREN}
export LHAPDF_DATA_PATH=${LHAPDF_DATA_PATH:-$(lhapdf-config --datadir 2> /dev/null)}
G4=${G4:-$SIREN_ROOT/geant4_shower/output/data/h5}
S=output/gamma_dis_samples

cd "$SIREN_ROOT/resources/analysis" || exit 1
echo "SIREN_ROOT=$SIREN_ROOT"
echo "LHAPDF_DATA_PATH=$LHAPDF_DATA_PATH"
echo "G4 showers: $G4"

FAILED=()
step() {
    echo
    echo "=== $(date '+%F %T')  $*"
    if ! "$@"; then
        echo "!!! FAILED: $*"
        FAILED+=("$*")
    fi
}

step python test_gamma_dis_conservation.py
step python test_gamma_dis_conservation.py --nc

if [ "${SAMPLES:-1}" = 1 ]; then
    step python gamma_dis_samples.py --which mono --n "$N_MONO"
    step python gamma_dis_ccnc_samples.py --which mono --n-mono "$N_CCNC"
    if [ "${SPECTRUM:-1}" = 1 ]; then
        step python gamma_dis_samples.py --which range --n "$N_SPECTRUM"
        step python gamma_dis_samples.py --which reweight --n "$N_SPECTRUM"
        step python gamma_dis_samples.py --which range --n "$N_SPECTRUM" --phys-index 1 --suffix _rev
        step python gamma_dis_samples.py --which reweight --n "$N_SPECTRUM" --phys-index 1 --test-index 2 --suffix _rev
    fi
fi

if [ "${LIBRARIES:-1}" = 1 ]; then
    step python pythia_dis_reference.py library --channel cc --n "$N_LIB" --out output/pythia_events_cc.npz
    step python pythia_dis_reference.py library --channel nc --n "$N_LIB" --out output/pythia_events_nc.npz
fi

if [ "${PLOTS:-1}" = 1 ]; then
    step python plot_gamma_dis_conservation.py --samples "$S"
    step python plot_gamma_dis_kinematics.py --samples "$S"
    if [ -e "$S/reweight_gamma.npz" ]; then
        step python plot_gamma_dis_reweighting.py --samples "$S"
    fi
    if [ -e "$S/reweight_gamma_rev.npz" ]; then
        step python plot_gamma_dis_reweighting.py --samples "$S" --suffix _rev
    fi
    if [ -e "$S/range_gamma.npz" ]; then
        step python plot_gamma_dis_showers.py --samples "$S" --sample range_gamma.npz
    fi
    step python plot_gamma_dis_ccnc.py --samples "$S" --g4-dir "$G4" --max-events 100000
    step python plot_gamma_dis_talk.py --samples "$S" --g4-dir "$G4" --enu 1e4 1e5 --n 300
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== $(date '+%F %T')  all steps finished"
else
    echo "=== $(date '+%F %T')  ${#FAILED[@]} step(s) failed:"
    printf '    %s\n' "${FAILED[@]}"
fi
echo "plots: $SIREN_ROOT/resources/analysis/output/gamma_dis_plots/"
