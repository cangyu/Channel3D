#!/bin/bash

#SBATCH --job-name=icoCh3Re40
#SBATCH --partition=cpu
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn

# CLEAN
wclean ../solver
rm -rf ../solver/PC3D
rm -rf processor*
rm -rf constant/*/polyMesh
rm -rf constant/polyMesh

# COMPILE
wmake ../solver

# PREPARE
blockMesh -region gas
decomposePar -region gas

# RUN
srun --mpi=pmi2 ../solver/PC3D -parallel

