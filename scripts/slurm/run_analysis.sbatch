#!/bin/bash
# SLURM launcher.  Partition, account and resources below are those of the cluster the
# paper runs used (SLAC S3DF): edit them for yours.

#SBATCH --partition=roma
#SBATCH --account=fermi:users
#SBATCH --job-name=crbreak
#SBATCH --output=outputs/output-%x-%j.txt
#SBATCH --error=errors/error-%x-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#
#SBATCH --time=0-120:00:00
#
#SBATCH --gpus 0

mkdir -p outputs errors

# Abort on any failure so we don't silently fall back to the wrong python.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Use the repo virtualenv if there is one, and this checkout of the package.
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Submit from the repository root with the analysis config (all options live there):
#   sbatch scripts/slurm/run_analysis.sbatch configs/baseline.yaml
CONFIG="${1:?usage: sbatch scripts/slurm/run_analysis.sbatch configs/<name>.yaml}"

# Resolve the output dir once so the logs land next to the plots.
OUTDIR="$(python3 -m crbreak.run "${CONFIG}" --print-outdir)"
mkdir -p "${OUTDIR}"
exec > "${OUTDIR}/output-${SLURM_JOB_NAME:-crbreak}-${SLURM_JOB_ID:-manual}.txt" \
     2> "${OUTDIR}/error-${SLURM_JOB_NAME:-crbreak}-${SLURM_JOB_ID:-manual}.txt"

python3 -u -m crbreak.run "${CONFIG}" --outdir "${OUTDIR}"
