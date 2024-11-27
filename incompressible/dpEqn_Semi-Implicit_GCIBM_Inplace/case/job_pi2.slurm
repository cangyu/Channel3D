#!/bin/bash

#SBATCH --job-name=iSR40
#SBATCH --partition=cpu
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn

# ENV
module load gcc/11.2.0
module load openmpi/4.1.5-gcc-11.2.0
module load eigen/3.4.0-gcc-11.2.0
module load openfoam-org/8-gcc-11.2.0

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
mpiexec -n 16 ../solver/PC3D -parallel
