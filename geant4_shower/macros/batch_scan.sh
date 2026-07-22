#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# batch_scan.sh
# Submit one SLURM job per (species, energy) pair.
# Skips jobs whose output file already exists.
#
# Usage:
#   bash batch_scan.sh           # submit only MISSING jobs (skips existing outputs)
#   bash batch_scan.sh --force   # RESUBMIT all jobs, overwriting existing outputs
#   bash batch_scan.sh --dryrun  # print commands without submitting
#   (flags can be combined, e.g. --force --dryrun)
#
# Run from the geant4_shower/ directory.
#
# All species, all energies 10 GeV – 30 TeV (8 points).
# 100 TeV deferred — will be run separately once checkpointing is added.
# Wall time: 3 days for all jobs (shared partition max — safe for all species).
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

PARTITION="shared"   # 3-day limit, no job-count cap
WALLTIME="3-00:00:00"

submit() {
    local PID=$1 NAME=$2 E=$3 NEVENTS=$4
    local OUTFILE="${OUTDIR}/shower_${NAME}_E${E}GeV.h5"
    local JOB="g4_${NAME}_E${E}"

    # Skip if output already exists, unless --force (resubmit / overwrite)
    if [[ -f "$OUTFILE" && "$FORCE" == false ]]; then
        echo "SKIP (exists): $OUTFILE"
        return
    fi

    local WRAP="source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh && \
          conda activate siren-dev && \
          $SIMBIN --pid $PID --energy $E --nevents $NEVENTS --output $OUTFILE"

    local CMD="sbatch \
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
}

ENERGIES=(10 30 100 300 1000 3000 10000 30000)

# ── Group 1: pip, pim, pi0, Kp, Km ──────────────────────────────────────────
echo "=== Group 1: pip pim pi0 Kp Km ==="
G1_PIDS=(  211  -211  111  321  -321)
G1_NAMES=( pip   pim  pi0   Kp    Km)

for i in "${!G1_PIDS[@]}"; do
    for E in "${ENERGIES[@]}"; do
        submit "${G1_PIDS[$i]}" "${G1_NAMES[$i]}" "$E" 1000
    done
done

# ── Group 2: KS, KL, p, n ────────────────────────────────────────────────────
echo "=== Group 2: KS KL p n ==="
G2_PIDS=(  310   130  2212  2112)
G2_NAMES=( KS    KL     p     n)

for i in "${!G2_PIDS[@]}"; do
    for E in "${ENERGIES[@]}"; do
        submit "${G2_PIDS[$i]}" "${G2_NAMES[$i]}" "$E" 1000
    done
done
