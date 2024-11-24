Foam::Info << "Create mesh for time = " << runTime.timeName() << Foam::nl << Foam::endl;

Foam::fvMesh mesh_gas
(
    Foam::IOobject("gas", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

const auto& bMesh_gas = mesh_gas.boundaryMesh();

Foam::globalMeshData meshInfo_gas(mesh_gas);

// Force calculation of extended edge addressing
{
    const Foam::labelListList& edgeFaces = mesh_gas.edgeFaces();
    const Foam::labelListList& edgeCells = mesh_gas.edgeCells();
    const Foam::labelListList& pointCells = mesh_gas.pointCells();
}
