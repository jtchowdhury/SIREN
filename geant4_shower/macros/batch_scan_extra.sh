#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# batch_scan_extra.sh
# Submit SLURM G4 jobs for the species that are NOT in the original hadron scan
# (macros/batch_scan.sh) but DO show up in the PYTHIA DIS final state:
#
#     em    e-    (PDG 11)     electromagnetic shower  (also covers e+ -> alias)
#     ph    gamma (PDG 22)     photon-initiated EM shower (~1 X0 deeper than e-)
#     pbar  p~    (PDG -2212)  antiproton (hadronic; annihilation included by FTFP_BERT)
#     nbar  n~    (PDG -2112)  antineutron (hadronic)
#
# Positrons are intentionally NOT simulated: an e+ shower is identical to an e-
# shower at these energies (end-of-track annihilation is negligible), so alias
# e+ -> em in the final-state mapping instead.
#
# The sim takes any PDG via --pid, so no C++ changes are needed. Output files are
# shower_<name>_E<E>GeV.h5; add each <name> to SPECIES in shower_gamma_model.py
# (and hadronic_showers.py) or load_g4_library will skip them.
#
# Usage (run from the geant4_shower/ directory):
#   bash macros/batch_scan_extra.sh            # submit only MISSING jobs
#   bash macros/batch_scan_extra.sh --force    # resubmit / overwrite
#   bash macros/batch_scan_extra.sh --dryrun   # print, don't submit
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

NEVENTS=1000
ENERGIES=(10 30 100 300 1000 3000 10000 30000)     # match the hadron grid

# species to simulate: PDG id  +  filename tag
PIDS=(   11    22   -2212  -2112)
NAMES=(  em    ph   pbar   nbar)

for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    NAME=${NAMES[$i]}
    for E in "${ENERGIES[@]}"; do
        OUTFILE="${OUTDIR}/shower_${NAME}_E${E}GeV.h5"
        JOB="g4_${NAME}_E${E}"

        if [[ -f "$OUTFILE" && "$FORCE" == false ]]; then
            echo "SKIP (exists): $OUTFILE"
            continue
        fi

        # sim has no baked-in RPATH, so the Geant4 libs (in the conda env) must be
        # on LD_LIBRARY_PATH at run time or the job dies loading shared libraries.
        WRAP="source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && \
              conda activate siren-dev && \
              export LD_LIBRARY_PATH=\\\$CONDA_PREFIX/lib:\\\$LD_LIBRARY_PATH && \
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
done
