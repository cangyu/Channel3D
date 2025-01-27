wmake ../solver

blockMesh -region gas

fluent3DMeshToFoam -scale 0.5e-3 ../mesh/fluent_solid.msh
cd constant
mkdir solid
mv polyMesh solid/
cd ..

decomposePar -region solid
decomposePar -region gas

mpiexec -n 4 ../solver/PC3D -parallel