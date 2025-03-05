/* Gas-phase */
Foam::Info << "\nCreate gas-phase mesh for time = " << runTime.timeName() << Foam::endl;

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

    Foam::Pout << edgeFaces.size() << " edgeFaces, " << edgeCells.size() << " edgeCells, " << pointCells.size() << " pointCells" << Foam::endl;
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

    Foam::Pout << edgeFaces.size() << " edgeFaces, " << edgeCells.size() << " edgeCells, " << pointCells.size() << " pointCells" << Foam::endl;
}

// Count and mark points on boundaries
Foam::boolList flag_pntOnBdry_solid(pointMesh_solid.size(), false);
Foam::boolList flag_pntOnParBdry_solid(pointMesh_solid.size(), false);
Foam::boolList flag_pntOnPhyBdry_solid(pointMesh_solid.size(), false);
{
    // Loop over all patches
    for (int i = 0; i < pointMesh_solid.boundary().size(); i++)
    {
        const auto &patch = pointMesh_solid.boundary()[i];
        const auto &pL = patch.meshPoints();

        for (int j = 0; j < pL.size(); j++)
            flag_pntOnBdry_solid[pL[j]] = true;

        if (patch.coupled())
        {
            for (int j = 0; j < pL.size(); j++)
                flag_pntOnParBdry_solid[pL[j]] = true;
        }
        else
        {
            for (int j = 0; j < pL.size(); j++)
                flag_pntOnPhyBdry_solid[pL[j]] = true;
        }
    }

    // Check
    int nParPnt = 0;
    for (int i = 0; i < pointMesh_solid.size(); i++)
    {
        if (flag_pntOnParBdry_solid[i])
            ++nParPnt;
    }
    const auto &slaves = pointMesh_solid.globalData().globalPointBoundaryCells();
    if (nParPnt != slaves.size())
    {
        Foam::Perr << "Inconsistent number of points on parallel boundaries: " << nParPnt << "(My) / " << slaves.size() << "(globalData)" << Foam::endl;
        return 1;
    }
}
