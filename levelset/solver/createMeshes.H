Foam::Info << "Create gas-phase mesh for time = " << runTime.timeName() << Foam::nl << Foam::endl;

Foam::fvMesh mesh_gas
(
    Foam::IOobject("gas", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

const auto &bMesh_gas = mesh_gas.boundaryMesh();

Foam::globalMeshData meshInfo_gas(mesh_gas);

Foam::pointMesh pointMesh_gas(mesh_gas);

// Force calculation of extended edge addressing
{
    const Foam::labelListList& edgeFaces = mesh_gas.edgeFaces();
    const Foam::labelListList& edgeCells = mesh_gas.edgeCells();
    const Foam::labelListList& pointCells = mesh_gas.pointCells();
}

Foam::Info << "Create solid-phase mesh for time = " << runTime.timeName() << Foam::nl << Foam::endl;

Foam::fvMesh mesh_solid
(
    Foam::IOobject("solid", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

const auto &bMesh_solid = mesh_solid.boundaryMesh();

Foam::globalMeshData meshInfo_solid(mesh_solid);

Foam::pointMesh pointMesh_solid(mesh_solid);

// Force calculation of extended edge addressing
{
    const Foam::labelListList& edgeFaces = mesh_solid.edgeFaces();
    const Foam::labelListList& edgeCells = mesh_solid.edgeCells();
    const Foam::labelListList& pointCells = mesh_solid.pointCells();
}
