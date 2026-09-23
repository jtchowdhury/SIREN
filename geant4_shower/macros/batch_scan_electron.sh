#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# batch_scan_electron.sh
# Submit one SLURM job per electron (e-) energy point, reusing the SAME sim
# binary, environment, partition and wall time as macros/batch_scan.sh.
# The Geant4 sim already accepts any PDG via --pid, so electrons need NO
# C++ changes: e- is PDG 11.
#
# Output files are named  shower_em_E<E>GeV.h5  so the model build picks them
# up as species "em" (you must add (11,"em") to SPECIES in shower_gamma_model.py
# first, otherwise load_g4_library skips them).
#
# Usage (run from the geant4_shower/ directory):
#   bash macros/batch_scan_electron.sh            # submit only MISSING jobs
#   bash macros/batch_scan_electron.sh --force    # resubmit / overwrite
#   bash macros/batch_scan_electron.sh --dryrun   # print, don't submit
#
# To reproduce the 1 TeV e- vs pi+ plot you only strictly need E=1000, but the
# full scan below gives electron a properly energy-interpolated model like the
# other species. Trim the ENERGIES array if you only want a couple of points.
# ─────────────────────────────────────────────────────────────────────────────

DRYRUN=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dryrun) DRYRUN=true ;;
        --force)  FORCE=true  ;;
    esac
done

SIMBIN="$(pwd)/build/sim"
OUTDIR="$(pwd)/output"
LOGDIR="$(pwd)/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

PARTITION="shared"      # same as batch_scan.sh (3-day limit, no job-count cap)
WALLTIME="3-00:00:00"

PID=11
NAME=em
NEVENTS=1000
#ENERGIES=(10 30 100 300 1000 3000 10000 30000)
ENERGIES=(1000)

for E in "${ENERGIES[@]}"; do
    OUTFILE="${OUTDIR}/shower_${NAME}_E${E}GeV.h5"
    JOB="g4_${NAME}_E${E}"

    if [[ -f "$OUTFILE" && "$FORCE" == false ]]; then
        echo "SKIP (exists): $OUTFILE"
        continue
    fi

    WRAP="source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && \
          conda activate siren-dev && \
          $SIMBIN --pid $PID --energy $E --nevents $NEVENTS --output $OUTFILE"

    CMD="sbatch \
        --job-name=${JOB} \
        --partition=${PARTITION} \
        --time=${WALLTIME} \
        --mem=4G \
        --cpus-per-task=1 \
        --output=${LOGDIR}/${JOB}_%j.out \
        --error=${LOGDIR}/${JOB}_%j.err \
        --wrap=\"${WRAP}\""

    if $DRYRUN; then
        echo "[DRYRUN] $CMD"
    else
        eval "$CMD"
        echo "Submitted: $JOB  (${NEVENTS} events, wall=${WALLTIME})"
    fi
done
