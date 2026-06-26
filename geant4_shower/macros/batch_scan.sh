#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# batch_scan.sh
# Submit one SLURM job per (species, energy) pair.
#
# Usage:
#   bash batch_scan.sh           # submits all 81 jobs
#   bash batch_scan.sh --dryrun  # prints commands without submitting
#
# Before running:
#   1. Adjust the module load lines below to match your HPC.
#   2. Set SIMBIN to the path of your compiled sim binary.
#   3. Run from the geant4_shower/ directory.
# ─────────────────────────────────────────────────────────────────────────────

DRYRUN=false
[[ "$1" == "--dryrun" ]] && DRYRUN=true

SIMBIN="$(pwd)/build/sim"
OUTDIR="$(pwd)/output"
LOGDIR="$(pwd)/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

# ── Particle species ──────────────────────────────────────────────────────────
PIDS=(  211  -211  111  321  -321  310  130  2212  2112)
NAMES=( pip   pim  pi0   Kp    Km   KS   KL     p     n)

# ── Kinetic energies [GeV] ────────────────────────────────────────────────────
# Range: 10 GeV to 100 TeV, roughly log-spaced.
# At TeV scale, kinetic energy ≈ total energy for all species here.
ENERGIES=(10 30 100 300 1000 3000 10000 30000 100000)

# ── Submit ────────────────────────────────────────────────────────────────────
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    NAME="${NAMES[$i]}"

    for E in "${ENERGIES[@]}"; do
        OUTFILE="${OUTDIR}/shower_${NAME}_E${E}GeV.h5"
        JOB="g4_${NAME}_E${E}"

        # Approximate wall-time: scales ~linearly with energy.
        # 100 GeV -> ~10 min, 100 TeV -> ~8 h. Adjust as needed.
        if   [[ $E -ge 30000 ]]; then WALLTIME="10:00:00"
        elif [[ $E -ge 3000  ]]; then WALLTIME="04:00:00"
        elif [[ $E -ge 300   ]]; then WALLTIME="02:00:00"
        else                          WALLTIME="00:30:00"
        fi

        # Activate conda env (brings Geant4, HDF5, and all data vars)
        WRAP="source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && \
              conda activate siren-dev && \
              $SIMBIN --pid $PID --energy $E --nevents 1000 --output $OUTFILE"

        CMD="sbatch \
            --job-name=${JOB} \
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
            echo "Submitted: $JOB  (wall $WALLTIME)"
        fi
    done
done
