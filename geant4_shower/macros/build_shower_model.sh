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
#SBATCH --time=02:00:00             # ~13 min expected on 16 cores; 2 h is safe margin
#SBATCH --output=shower_fit_%j.log

# ---- ADJUST THESE TWO FOR YOUR CLUSTER (partition / account names) ----
# #SBATCH --partition=shared
# #SBATCH --account=your_account

# ---- environment (matches your interactive setup) ----
source ~/.bashrc
conda activate siren-dev            # <-- your conda env

# ---- run from this script's directory ----
cd "$(dirname "$0")"

echo "Starting build on $SLURM_CPUS_PER_TASK cores at $(date)"
python shower_gamma_model.py \
    --g4-dir ../output \
    --n-jobs "$SLURM_CPUS_PER_TASK" \
    --save ../output/shower_model.pkl \
    --no-validate
echo "Done at $(date)"
