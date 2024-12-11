wmake ../solver

blockMesh -region gas

decomposePar -region gas

mpiexec -n 4 ../solver/PC3D -parallel
