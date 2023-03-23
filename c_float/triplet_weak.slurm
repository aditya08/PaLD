#!/bin/bash -l
#SBATCH --partition=small
#SBATCH --job-name="triplet_weakscale_job_cores24"
#SBATCH --time=0-02:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=devaraa@wfu.edu
#SBATCH --account="devarakondaGrp"
#SBATCH --mem=96gb
#SBATCH --output="triplet_weakscaling_cores24.out"
#SBATCH --error="triplet_weakscaling_cores24.err"
#SBATCH --constraint="cascade"

cd /home/devaraa/PaLD/c_float
./triplet_weak_scaling.sh 32
