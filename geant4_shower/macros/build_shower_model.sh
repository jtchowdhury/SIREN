#!/bin/bash
# =====================================================================
# Build the gamma shower model from the G4 profile library, in parallel,
# and save it to one .pkl file.  Run this ONCE as a batch job.
#
# Submit with:   sbatch build_shower_model.sh
# Watch it with: tail -f shower_fit_<jobid>.log
# =====================================================================
#SBATCH --job-name=shower_fit
#SBATCH --cpus-per-task=16          # number of parallel fit workers
#SBATCH --mem=16G
#SBATCH --time=02:00:00             # ~12 min expected on 16 cores; 2 h is safe margin
#SBATCH --output=shower_fit_%j.log
#SBATCH --partition=shared          # same partition your G4 jobs use (3-day limit)

# ---- environment: identical to your G4 jobs (macros/batch_scan.sh). Do NOT
#      source ~/.bashrc here -- that loads the gcc module (libmpfr error) and
#      triggers your auto-upload. Sourcing conda.sh directly avoids both. ----
source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate siren-dev

# ---- cd into the repo. SLURM runs a SPOOLED copy of this script, so "$0" is
#      NOT macros/ -- that was the original bug. The paths below are absolute
#      anyway, so this just keeps the working directory sane. ----
cd "$HOME/SIREN/geant4_shower"

echo "Starting build on $SLURM_CPUS_PER_TASK cores at $(date)"
python /n/home13/jchowdhury/SIREN/geant4_shower/python_scripts/shower_gamma_model.py \
    --g4-dir /n/home13/jchowdhury/SIREN/geant4_shower/output \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --save /n/home13/jchowdhury/SIREN/geant4_shower/output/shower_model.pkl \
    --no-validate
echo "Done at $(date)"
