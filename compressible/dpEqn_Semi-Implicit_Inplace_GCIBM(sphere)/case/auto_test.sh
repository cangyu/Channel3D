wmake ../solver

blockMesh -region gas

decomposePar -region gas

mpiexec -n 16 ../solver/PC3D -parallel
