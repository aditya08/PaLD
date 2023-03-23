#!/bin/bash -l
#SBATCH --partition=small
#SBATCH --job-name="triplet_strongscale_job3"
#SBATCH --time=0-04:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=devaraa@wfu.edu
#SBATCH --account="devarakondaGrp"
#SBATCH --mem=96gb
#SBATCH --output="triplet_strongscaling_3.out"
#SBATCH --error="triplet_strongscaling_3.err"
#SBATCH --constraint="cascade"

cd /home/devaraa/PaLD/c_float
./triplet_strong_scaling.sh 8192 256
