/* Gas-phase */
Foam::Info << "Create gas-phase mesh for time = " << runTime.timeName() << Foam::endl;

Foam::fvMesh mesh_gas
(
    Foam::IOobject("gas", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

Foam::pointMesh pointMesh_gas(mesh_gas);

// Force calculation of extended edge addressing
{
    const Foam::labelListList& edgeFaces = mesh_gas.edgeFaces();
    const Foam::labelListList& edgeCells = mesh_gas.edgeCells();
    const Foam::labelListList& pointCells = mesh_gas.pointCells();
    Foam::Pout << edgeFaces.size() << " edgeFaces, "
               << edgeCells.size() << " edgeCells, "
               << pointCells.size() << " pointCells"
               << Foam::endl;
}

/* Solid-phase */
Foam::Info << "\nCreate solid-phase mesh for time = " << runTime.timeName() << Foam::endl;

Foam::fvMesh mesh_solid
(
    Foam::IOobject("solid", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

Foam::pointMesh pointMesh_solid(mesh_solid);

// Force calculation of extended edge addressing
{
    const Foam::labelListList& edgeFaces = mesh_solid.edgeFaces();
    const Foam::labelListList& edgeCells = mesh_solid.edgeCells();
    const Foam::labelListList& pointCells = mesh_solid.pointCells();
    Foam::Pout << edgeFaces.size() << " edgeFaces, "
               << edgeCells.size() << " edgeCells, "
               << pointCells.size() << " pointCells"
               << Foam::endl;
}
