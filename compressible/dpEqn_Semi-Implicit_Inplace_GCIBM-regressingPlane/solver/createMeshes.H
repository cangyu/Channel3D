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

    Foam::Pout << edgeFaces.size() << " edgeFaces, " << edgeCells.size() << " edgeCells, " << pointCells.size() << " pointCells" << Foam::endl;
}

// Test globalMeshData
{
    const Foam::globalMeshData &globalData = mesh_gas.globalData();

    const Foam::indirectPrimitivePatch &coupledPatch = globalData.coupledPatch();
    const Foam::globalIndexAndTransform &transforms = globalData.globalTransforms();
    const Foam::labelList &boundaryCells = globalData.boundaryCells();
    const Foam::mapDistribute &globalPointBoundaryCellsMap = globalData.globalPointBoundaryCellsMap();
    const Foam::labelListList &slaves = globalData.globalPointBoundaryCells();
    const Foam::labelListList &transformedSlaves = globalData.globalPointTransformedBoundaryCells();

    Foam::Pout << "nPoints=" << mesh_gas.nPoints()
               << ", constructSize=" << globalPointBoundaryCellsMap.constructSize()
               << ", boundaryCells.size=" << boundaryCells.size()
               << ", slaves.size=" << slaves.size()
               << ", coupledPatch.meshPoints.size=" << coupledPatch.meshPoints().size()
               << Foam::endl;

    // Collect
    Foam::vectorList bData(globalPointBoundaryCellsMap.constructSize());
    for (int i = 0; i < boundaryCells.size(); i++)
    {
        const auto cI = boundaryCells[i];
        bData[i] = mesh_gas.C()[cI]/h;
    }

    std::ofstream fout(std::to_string(Foam::Pstream::myProcNo())+"_bData_before_exchange.txt");
    for (int i = 0; i < bData.size(); i++)
    {
        fout << i << "(" << bData[i].x() << "," << bData[i].y() << "," << bData[i].z() << ")" << std::endl;
    }
    fout.close();

    // Exchange data
    globalPointBoundaryCellsMap.distribute
    (
        transforms,
        bData,
        Foam::mapDistribute::transformPosition()
    );

    fout.open(std::to_string(Foam::Pstream::myProcNo())+"_bData_after_exchange.txt");
    for (int i = 0; i < bData.size(); i++)
    {
        fout << i << "(" << bData[i].x() << "," << bData[i].y() << "," << bData[i].z() << ")" << std::endl;
    }
    fout.close();

    // globalPointBoundaryCellsMap.reverseDistribute(bData.size(), bData);

    Foam::List<Foam::vectorList> dst(mesh_gas.nPoints());
    int nEmpty = 0;
    for (int i = 0; i < slaves.size(); i++)
    {
        const Foam::labelList &pointCells = slaves[i];
        if (pointCells.size() == 0)
        {
            ++nEmpty;
            continue;
        }

        const Foam::label pI = coupledPatch.meshPoints()[i];
        if (dst[pI].size() != pointCells.size())
            dst[pI].resize(pointCells.size());

        for (int j = 0; j < pointCells.size(); j++)
            dst[pI][j] = bData[pointCells[j]];
    }

    // globalPointBoundaryCellsMap.reverseDistribute(dst.size(), dst);

    // globalData.syncPointData
    // (
    //     dst,
    //     Foam::plusEqOp<Foam::vectorList>(),
    //     Foam::mapDistribute::transform()
    // );

    // Foam::globalMeshData::syncData
    // (
    //     dst,
    //     slaves,
    //     transformedSlaves,
    //     globalPointBoundaryCellsMap,
    //     Foam::cut::appendOp<Foam::vectorList>()
    // );

    Foam::Pout << "nEmpty=" << nEmpty << Foam::endl;

    // Compare
    for (int i = 0; i < mesh_gas.boundaryMesh().size(); i++)
    {
        const Foam::polyPatch &patch = mesh_gas.boundaryMesh()[i];
    	// Foam::Pout << patch.name() << ": coupled=" << patch.coupled() << Foam::endl;
        if (!patch.coupled())
            continue;

        std::ofstream fout(patch.name()+"_pointLabel.txt");
        for (int j = 0; j < patch.meshPoints().size(); j++)
        {
            const auto pI = patch.meshPoints()[j];
            const auto &P = mesh_gas.points()[pI];
            const auto &pCL = mesh_gas.pointCells()[pI];

            fout << pI << "  (" << P.x()/h << "," << P.y()/h << "," << P.z()/h << ")" 
                 << "  " << pCL.size() << "/" << dst[pI].size();
            
            for (int k = 0; k < dst[pI].size(); k++)
                fout << " [" << k << "](" << dst[pI][k].x()/h << "," << dst[pI][k].y()/h << "," << dst[pI][k].z()/h << ")";
            
            fout << std::endl;
        }
        fout.close();
    }
    
    return 188;
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
