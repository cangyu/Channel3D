#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "fvc.H"
#include "fvm.H"
#include "timeSelector.H"
#include "labelIOField.H"
#include "labelFieldIOField.H"
#include "labelList.H"
#include "scalarIOField.H"
#include "scalarFieldIOField.H"
#include "vectorIOField.H"
#include "vectorFieldIOField.H"
#include "vectorList.H"
#include "tensorIOField.H"
#include "tensorFieldIOField.H"
#include "tensorList.H"
#include "pointFields.H"
#include "centredCECCellToCellStencilObject.H"
#include "centredCFCCellToCellStencilObject.H"
#include "centredCPCCellToCellStencilObject.H"
#include "fluidThermo.H"
#include "fluidThermoMomentumTransportModel.H"
#include "fluidThermophysicalTransportModel.H"
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include <limits>
#include <string>
#include <Eigen/Dense>

/* Constants */
const Foam::scalar s2ns = 1e9, ns2s = 1.0/s2ns;
const Foam::scalar s2us = 1e6, us2s = 1.0/s2us;
const Foam::scalar s2ms = 1e3, ms2s = 1.0/s2ms;
const Foam::scalar m2nm = 1e9, nm2m = 1.0/m2nm;
const Foam::scalar m2um = 1e6, um2m = 1.0/m2um;
const Foam::scalar m2mm = 1e3, mm2m = 1.0/m2mm;
const Foam::scalar m2cm = 1e2, cm2m = 1.0/m2cm;
const Foam::scalar kcal2J = 4186.8;
const Foam::scalar one_atm = 101325.0;           // Unit: Pa
const Foam::scalar one_bar = 100000.0;           // Unit: Pa
const Foam::scalar one_psi = 6894.76;            // Unit: Pa
const Foam::scalar one_mpa = 1e6;                // Unit: Pa
const Foam::scalar G0 = 1.4;                     // Specific heat ratio for ideal gas

// Universial gas constant, Unit: J/mol/K
const Foam::dimensionedScalar R(Foam::dimEnergy/Foam::dimMoles/Foam::dimTemperature, 8.31446261815324);

// Molecular weight, Unit: kg/mol
const Foam::dimensionedScalar MW(Foam::dimMass/Foam::dimMoles, 26e-3);

// Gas constant, Unit: J/kg/K
const Foam::dimensionedScalar Rg(R/MW);

/* Properties of the solid material */
const double rho_AP = 1950.0;                    // Density of AP, Unit: kg/m^3
const double rho_HTPB = 920.0;                   // Density of HTPB, Unit: kg/m^3
const double c_AP = 0.3 * kcal2J;                // Heat capacity of AP, Unit: J/kg/K
const double c_HTPB = 0.3 * kcal2J;              // Heat capacity of HTPB, Unit: J/kg/K
const double lambda_AP = 0.405;                  // Conductivity of AP, Unit: W/m/K
const double lambda_HTPB = 0.276;                // Conductivity of HTPB, Unit: W/m/K
const double qL_AP = -80 * kcal2J;               // Latent heat of AP, Unit: J/kg
const double qL_HTPB = -66 * kcal2J;             // Latent heat of HTPB, Unit: J/kg

/* Cartesian grid */
const Foam::scalar L = 0.5*mm2m;                 // Domain characteristic length, Unit: m
const Foam::label N = 32;                        // Grid resolution
const Foam::scalar h = L / 32;                   // Spacing
const Foam::scalar h_inv = 1.0 / h;
const Foam::scalar xMin = 0.0, xMax = xMin+L;    // Range in X-direction
const Foam::scalar yMin = 0.0, yMax = yMin+L;    // Range in Y-direction
const Foam::scalar zMin = 0.0, zMax = zMin+2*L;  // Range in Z-direction

/* Classification of IBM cells */
const Foam::scalar cFluid = 1.0;                 // Fluid cell
const Foam::scalar cGhost = 0.0;                 // Ghost cell
const Foam::scalar cSolid = -1.0;                // Solid cell

/* Flow condition */
const Foam::scalar Pr = 0.71;                    // Prandtl number
const Foam::scalar p0 = 2.07 * one_mpa;          // Ambient pressure
const Foam::scalar T0 = 300.0;                   // Initial temperature

/* Plane param */
const Foam::scalar plane_z = 0.25*mm2m;          // Vertical position at initial
const Foam::scalar plane_T = 300.0;              // Temperature

/**
 * Check if two scalars are equal in the float-point-number sense.
 * The relative error is set to 1e-5.
 * If "x" or "y" is 0, better to use the specialized version.
 */
inline bool isEqual(double x, double y)
{
    return std::islessequal(std::abs(x-y), 1e-5 * std::abs(x));
}

/**
 * Check if the given scalar is zero in the float-point-number sense.
 * The absolute error is set to 1e-7.
 */
inline bool isZero(double x)
{
    return std::isless(std::abs(x), 1e-7);
}

/**
 * Check the type of grid cells, based on the prescribed marker value.
 * These are helper functions to avoid explicit float-point comparsion.
 */
inline bool isFluidCell(double marker)
{
    return isEqual(marker, cFluid);
}

inline bool isSolidCell(double marker)
{
    return isEqual(marker, cSolid);
}

inline bool isGhostCell(double marker)
{
    return isZero(marker);
}

/**
 * Identify the position on equidistant grid.
 * @param x_ The target coordinate.
 * @param x0_ The starting coordinate.
 * @param h_ The spacing of the equidistant grid.
 */
inline bool atFullSpacing(double x_, double x0_, double h_)
{
    const double idx = (x_ - x0_) / h_;
    const double res = std::fmod(idx, 1.0);
    return isZero(res) || isEqual(res, 1.0);
}

inline bool atHalfSpacing(double x_, double x0_, double h_)
{
    const double idx = (x_ - x0_) / h_;
    const double res = std::fmod(idx, 1.0);
    return isEqual(res, 0.5);
}

/**
 * Check if the three-dimensional point is located on any cell centroid of the Cartesian grid.
 * Parameters of the Cartesian grid are prescribed at the beginning.
 */
inline bool isCellCentroid(const Foam::vector &p)
{
    return atHalfSpacing(p.x(), xMin, h) && atHalfSpacing(p.y(), yMin, h) && atHalfSpacing(p.z(), zMin, h);
}

/**
 * Check if the three-dimensional point is located on any node of the Cartesian grid.
 * Parameters of the Cartesian grid are prescribed at the beginning.
 */
inline bool isNode(const Foam::vector &p)
{
    return atFullSpacing(p.x(), xMin, h) && atFullSpacing(p.y(), yMin, h) && atFullSpacing(p.z(), zMin, h);
}

/**
 * On the Cartesian grid, convert the three-dimensional coordinate to the index of each dimension.
 * Parameters of the Cartesian grid are prescribed at the beginning.
 */
inline void xyz2ijk(const Foam::vector &p_, int &i_, int &j_, int &k_)
{
    i_ = static_cast<int>(std::fma(p_.x() - xMin, h_inv, 0.5));
    j_ = static_cast<int>(std::fma(p_.y() - yMin, h_inv, 0.5));
    k_ = static_cast<int>(std::fma(p_.z() - zMin, h_inv, 0.5));
}

/**
 * dst = src1 * src2
 * The boundary value of "src1" and "src2" are prescribed by boundary conditions.
 */
void setBdryVal(const Foam::fvMesh &mesh, const Foam::volScalarField &src1, const Foam::volVectorField &src2, Foam::surfaceScalarField &dst)
{
    for (int pI = 0; pI < mesh.boundary().size(); pI++)
    {
        const auto &curPatch = mesh.boundary()[pI];
        const auto &patch_val1 = src1.boundaryField()[pI];
        const auto &patch_val2 = src2.boundaryField()[pI];
        const auto &patch_Sn = mesh.Sf().boundaryField()[pI];
        auto &patch_dst = dst.boundaryFieldRef()[pI];
        for (int fI = 0; fI < curPatch.size(); fI++)
            patch_dst[fI] = patch_val1[fI] * patch_val2[fI] & patch_Sn[fI];
    }
}

/**
 * dst = src
 * The boundary value of "src" is prescribed by boundary condition.
 */
void setBdryVal(const Foam::fvMesh &mesh, const Foam::volVectorField &src, Foam::surfaceScalarField &dst)
{
    for (int pI = 0; pI < mesh.boundary().size(); pI++)
    {
        const auto &curPatch = mesh.boundary()[pI];
        const auto &patch_val = src.boundaryField()[pI];
        const auto &patch_Sn = mesh.Sf().boundaryField()[pI];
        auto &patch_dst = dst.boundaryFieldRef()[pI];
        for (int fI = 0; fI < curPatch.size(); fI++)
            patch_dst[fI] = patch_val[fI] & patch_Sn[fI];
    }
}

/**
 * Check the range of the target field with mask.
 * @param src Target scalar field.
 * @param flag Mask of the "src".
 * @param norm1 The L1 norm of "src".
 * @param norm2 The L2 norm of "src".
 * @param normInf The infinity norm of "src".
 * @param minVal Minimum of "src".
 * @param maxVal Maximum of "src".
 */
void diagnose(const Foam::fvMesh &mesh, const Foam::volScalarField &src, const Foam::volScalarField &flag, double &norm1, double &norm2, double &normInf, double &minVal, double &maxVal)
{
    const Foam::globalMeshData &meshInfo = mesh.globalData();
    int nMaskedCell = 0;
    norm1 = norm2 = normInf = 0.0;
    minVal = std::numeric_limits<double>::max();
    maxVal = std::numeric_limits<double>::min();
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isZero(flag[i]))
        {
            ++nMaskedCell;
            continue;
        }

        const double cVal = std::abs(src[i]);
        normInf = std::max(normInf, cVal);
        norm1 += cVal;
        norm2 += cVal*cVal;
        minVal = std::min(minVal, src[i]);
        maxVal = std::max(maxVal, src[i]);
    }
    Foam::reduce(minVal, Foam::minOp<Foam::scalar>());
    Foam::reduce(maxVal, Foam::maxOp<Foam::scalar>());
    Foam::reduce(normInf, Foam::maxOp<Foam::scalar>());
    Foam::reduce(norm1, Foam::sumOp<Foam::scalar>());
    Foam::reduce(norm2, Foam::sumOp<Foam::scalar>());
    Foam::reduce(nMaskedCell, Foam::sumOp<int>());
    const int nActiveCell = meshInfo.nTotalCells() - nMaskedCell;
    norm1 /= nActiveCell;
    norm2 = std::sqrt(norm2/nActiveCell);
}

void diagnose(const Foam::fvMesh &mesh, const Foam::volScalarField &src, const Foam::volScalarField &flag, double &norm1, double &norm2, double &normInf)
{
    const Foam::globalMeshData &meshInfo = mesh.globalData();
    int nMaskedCell = 0;
    norm1 = norm2 = normInf = 0.0;
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isZero(flag[i]))
        {
            ++nMaskedCell;
            continue;
        }

        const double cVal = std::abs(src[i]);
        normInf = std::max(normInf, cVal);
        norm1 += cVal;
        norm2 += cVal*cVal;
    }
    Foam::reduce(normInf, Foam::maxOp<Foam::scalar>());
    Foam::reduce(norm1, Foam::sumOp<Foam::scalar>());
    Foam::reduce(norm2, Foam::sumOp<Foam::scalar>());
    Foam::reduce(nMaskedCell, Foam::sumOp<int>());
    const int nActiveCell = meshInfo.nTotalCells() - nMaskedCell;
    norm1 /= nActiveCell;
    norm2 = std::sqrt(norm2/nActiveCell);
}

void diagnose(const Foam::fvMesh &mesh, const Foam::volScalarField &src, const Foam::volScalarField &flag, double &minVal, double &maxVal)
{
    minVal = std::numeric_limits<double>::max();
    maxVal = std::numeric_limits<double>::min();
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isZero(flag[i]))
            continue;

        minVal = std::min(minVal, src[i]);
        maxVal = std::max(maxVal, src[i]);
    }
    Foam::reduce(minVal, Foam::minOp<Foam::scalar>());
    Foam::reduce(maxVal, Foam::maxOp<Foam::scalar>());
}

/**
 * Collect the data on cells surrounding each point.
 * Points on parallel boundaries are handled.
 */
int collectData_pointCell(const Foam::pointMesh &pointMesh, const Foam::boolList &pntCoupledFlag, const Foam::volVectorField &src, Foam::List<Foam::vectorList> &dst)
{
    const Foam::polyMesh &polyMesh = pointMesh.mesh();
    const Foam::globalMeshData &globalData = pointMesh.globalData();

    const Foam::indirectPrimitivePatch &coupledPatch = globalData.coupledPatch();
    const Foam::globalIndexAndTransform &transforms = globalData.globalTransforms();
    const Foam::labelList &boundaryCells = globalData.boundaryCells();
    const Foam::mapDistribute &globalPointBoundaryCellsMap = globalData.globalPointBoundaryCellsMap();
    const Foam::labelListList &slaves = globalData.globalPointBoundaryCells();

    /* Collect data for points inside the domain and on physical boundaries */
    dst.resize(pointMesh.size());
    for (int i = 0; i < pointMesh.size(); i++)
    {
        // if (pntCoupledFlag[i])
        //     continue;

        const Foam::labelList &cL = polyMesh.pointCells()[i];
        dst[i].resize(cL.size());
        for (int j = 0; j < cL.size(); j++)
            dst[i][j] = src[cL[j]];
    }

    /* Collect data for points on parallel boundaries */
    // Record local data
    Foam::vectorList bData(globalPointBoundaryCellsMap.constructSize());
    for (int i = 0; i < boundaryCells.size(); i++)
    {
        const auto cI = boundaryCells[i];
        bData[i] = src[cI];
    }
    // Exchange data
    globalPointBoundaryCellsMap.distribute
    (
        transforms,
        bData,
        Foam::mapDistribute::transformPosition()
    );
    // Store data in order
    for (int i = 0; i < slaves.size(); i++)
    {
        const Foam::labelList &pointCells = slaves[i];
        const Foam::label pI = coupledPatch.meshPoints()[i];

        if (dst[pI].size() != pointCells.size())
        {
            Foam::Pout << "point[" << pI << "]: " << dst[pI].size() << "(local)/" << pointCells.size() << "(global)" << Foam::endl;
            dst[pI].resize(pointCells.size());
        }
        for (int j = 0; j < pointCells.size(); j++)
            dst[pI][j] = bData[pointCells[j]];

        // if (pntCoupledFlag[pI] && dst[pI].empty())
        // {
            
        // }
        // else
        // {
        //     Foam::Perr << "Problem detected on parallel-boundary point " << i << "(" << pI << "): "
        //                << "\tpntCoupledFlag=" << pntCoupledFlag[pI] << ", "
        //                << "\tempty=" << dst[pI].empty()
        //                << Foam::endl;
        //     return 2;
        // }
    }

    return 0;
}

/**
 * Check if the given LEVEL-SET value indicates the FLUID phase.
 */
inline bool inFluid(Foam::scalar c)
{
    return c > 0.0;
}

/**
 * Check if the given LEVEL-SET value indicates the SOLID phase.
 */
inline bool inSolid(Foam::scalar c)
{
    return c < 0.0;
}

/**
 * Check if the given LEVEL-SET value is near the INTERFACE (zero-contour of the level-set function).
 * The criteria is set to 1/10 of the Cartesian grid spacing.
 */
inline bool nearSurf(Foam::scalar c)
{
    return std::abs(c) < 0.1 * h;
}

/**
 * Check if the mesh cell is intersected by the iso-surface.
 * Based on its nodal LEVEL-SET values.
 */
bool cellIsIntersected(const Foam::scalarList &v)
{
    const bool flag = inSolid(v[0]);
    for (int i = 1; i < v.size(); i++)
    {
        if (inSolid(v[i]) != flag)
            return true;
    }
    return false;
}

/**
 * Classify the grid cell based on its LEVEL-SET values.
 * @param c LEVEL-SET value at the cell centroid.
 * @param v LEVEL-SET values on vertexes.
 * @return Tag of the Ghost-cell Immersed-Boundary Method. (Fluid/Solid/Ghost)
 */
Foam::scalar cellNum(const Foam::scalar c, const Foam::scalarList &v)
{
    if (inFluid(c))
        return cFluid;
    else
    {
        if (cellIsIntersected(v))
            return cGhost;
        else
        {
            bool awayFromSruf = true;
            for (int i = 0; i < v.size(); i++)
            {
                if (nearSurf(v[i]))
                {
                    awayFromSruf = false;
                    break;
                }
            }

            if (awayFromSruf)
                return cSolid;
            else
                return cGhost;
        }
    }
}

/**
 * Update the IBM tag of all Cartesian grid cells.
 * @param mesh The target Cartesian grid.
 * @param phi LEVEL-SET field on vertexes.
 * @param marker Store the tag of each grid cell.
 */
void identifyIBCell(const Foam::fvMesh &mesh, const Foam::pointScalarField &phi, Foam::volScalarField &marker)
{
    int nSolid = 0, nGhost = 0, nFluid = 0;
    for (int i = 0; i < mesh.nCells(); i++)
    {
        const auto &pL = mesh.cellPoints()[i];

        Foam::scalarList val(pL.size());
        Foam::scalar val_mean = 0.0;
        for (int j = 0; j < pL.size(); j++)
        {
            const auto pI = pL[j];
            val[j] = phi[pI];
            val_mean += val[j];
        }
        val_mean /= pL.size();

        marker[i] = cellNum(val_mean, val);

        if (isSolidCell(marker[i]))
            ++nSolid;
        else if (isFluidCell(marker[i]))
            ++nFluid;
        else
            ++nGhost;
    }

    // Ensure all solid cells not adjacent to fluid cells directly (through face)
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isSolidCell(marker[i]))
        {
            const auto &cC = mesh.cellCells()[i];
            for (int j = 0; j < cC.size(); j++)
            {
                const auto adjCI = cC[j];
                if (isFluidCell(marker[adjCI]))
                {
                    marker[adjCI] = cGhost;
                    --nFluid;
                    ++nGhost;
                }
            }
        }
    }

    if (nSolid > 0)
    {
        // Remove redundent ghost cells on fluid side (through face connection)
        for (int i = 0; i < mesh.nCells(); i++)
        {
            if (isGhostCell(marker[i]))
            {
                const auto &cC = mesh.cellCells()[i];
                bool adjToSolid = false;
                for (int j = 0; j < cC.size(); j++)
                {
                    const auto adjCI = cC[j];
                    if (isSolidCell(marker[adjCI]))
                    {
                        adjToSolid = true;
                        break;
                    }
                }
                if (!adjToSolid)
                {
                    marker[i] = cFluid;
                    --nGhost;
                    ++nFluid;
                }
            }
        }
    }

    if (nFluid > 0)
    {
        // Remove redundent ghost cells on solid side (through face connection)
        for (int i = 0; i < mesh.nCells(); i++)
        {
            if (isGhostCell(marker[i]))
            {
                const auto &cC = mesh.cellCells()[i];
                bool adjToFluid = false;
                for (int j = 0; j < cC.size(); j++)
                {
                    const auto adjCI = cC[j];
                    if (isFluidCell(marker[adjCI]))
                    {
                        adjToFluid = true;
                        break;
                    }
                }
                if (!adjToFluid)
                {
                    marker[i] = cSolid;
                    --nGhost;
                    ++nSolid;
                }
            }
        }
    }

    Foam::Pout << nFluid << "(fluid) + " << nGhost << "(ghost) + " << nSolid << "(solid) = " << nFluid+nGhost+nSolid << "/" << mesh.nCells() << Foam::endl;
}

/**
 * Immersed-Bounary Method interpolation routines.
 */
void ibInterp_Dirichlet_Linear(const Foam::vector &p_src, Foam::scalar val_src, const std::vector<Foam::vector> &p_adj, const std::vector<Foam::scalar> &val_adj, const Foam::vector &p_dst, Foam::scalar &val_dst)
{
    const int nADJ = p_adj.size();
    double dx, dy, dz;

    Eigen::MatrixXd A(nADJ, 3);
    Eigen::VectorXd b(nADJ);
    Eigen::VectorXd C(3);

    // Assemble adjacent interpolation equations
    for (int i = 0; i < nADJ; i++)
    {
        b[i] = val_adj[i] - val_src;

        dx = p_adj[i].x() - p_src.x();
        dy = p_adj[i].y() - p_src.y();
        dz = p_adj[i].z() - p_src.z();

        A(i, 0) = dx;
        A(i, 1) = dy;
        A(i, 2) = dz;
    }

    // Solve least-squares coefficients
    C = A.colPivHouseholderQr().solve(b);

    // Interpolate at the image point
    const Foam::vector p_img = 2.0 * p_src - p_dst;
    dx = p_img.x() - p_src.x();
    dy = p_img.y() - p_src.y();
    dz = p_img.z() - p_src.z();
    const Foam::scalar val_img = val_src + C[0] * dx + C[1] * dy + C[2] * dz;

    // Interpolate at the ghost point
    val_dst = 2.0 * val_src - val_img;
}

void ibInterp_Neumann_Linear(const Foam::vector &p_src, const Foam::vector &n_src, Foam::scalar snGrad_src, const std::vector<Foam::vector> &p_adj, const std::vector<Foam::scalar> &val_adj, const Foam::vector &p_dst, Foam::scalar &val_dst)
{
    const int nADJ = p_adj.size();
    double dx, dy, dz;

    Eigen::MatrixXd A(nADJ+1, 4);
    Eigen::VectorXd b(nADJ+1);
    Eigen::VectorXd C(4);
    
    // Assemble adjacent interpolation equations
    for (int i = 0; i < nADJ; i++)
    {
        b[i] = val_adj[i];

        dx = p_adj[i].x() - p_src.x();
        dy = p_adj[i].y() - p_src.y();
        dz = p_adj[i].z() - p_src.z();

        A(i, 0) = 1.0;
        A(i, 1) = dx;
        A(i, 2) = dy;
        A(i, 3) = dz;
    }

    // Incoporate the boundary condition
    b[nADJ] = snGrad_src;
    A(nADJ, 0) = 0.0;
    A(nADJ, 1) = n_src.x();
    A(nADJ, 2) = n_src.y();
    A(nADJ, 3) = n_src.z();

    // Solve least-squares coefficients
    C = A.colPivHouseholderQr().solve(b);

    // Interpolate at the ghost point
    dx = p_dst.x() - p_src.x();
    dy = p_dst.y() - p_src.y();
    dz = p_dst.z() - p_src.z();
    val_dst = C[0] + C[1] * dx + C[2] * dy + C[3] * dz;
}

void ibInterp_zeroGradient_Linear(const Foam::vector &p_src, const Foam::vector &n_src, const std::vector<Foam::vector> &p_adj, const std::vector<Foam::scalar> &val_adj, const Foam::vector &p_dst, Foam::scalar &val_dst)
{
    const int nADJ = p_adj.size();
    double dx, dy, dz;

    Eigen::MatrixXd A(nADJ+1, 4);
    Eigen::VectorXd b(nADJ+1);
    Eigen::VectorXd C(4);
    
    // Assemble adjacent interpolation equations
    for (int i = 0; i < nADJ; i++)
    {
        b[i] = val_adj[i];

        dx = p_adj[i].x() - p_src.x();
        dy = p_adj[i].y() - p_src.y();
        dz = p_adj[i].z() - p_src.z();

        A(i, 0) = 1.0;
        A(i, 1) = dx;
        A(i, 2) = dy;
        A(i, 3) = dz;
    }

    // Incoporate the boundary condition
    b[nADJ] = 0.0;
    A(nADJ, 0) = 0.0;
    A(nADJ, 1) = n_src.x();
    A(nADJ, 2) = n_src.y();
    A(nADJ, 3) = n_src.z();

    // Solve least-squares coefficients
    C = A.colPivHouseholderQr().solve(b);

    // Interpolate at the image point
    const Foam::vector p_img = 2.0 * p_src - p_dst;
    dx = p_img.x() - p_src.x();
    dy = p_img.y() - p_src.y();
    dz = p_img.z() - p_src.z();
    const Foam::scalar val_img = C[0] + C[1] * dx + C[2] * dy + C[3] * dz;

    // Interpolate at the ghost point
    val_dst = val_img;
}

void ibInterp_Robin_Linear(const Foam::vector &p_src, const Foam::vector &n_src, Foam::scalar bc_alpha, Foam::scalar bc_beta, Foam::scalar bc_gamma, const std::vector<Foam::vector> &p_adj, const std::vector<Foam::scalar> &val_adj, const Foam::vector &p_dst, Foam::scalar &val_dst)
{
    const int nADJ = p_adj.size();
    double dx, dy, dz;

    Eigen::MatrixXd A(nADJ+1, 4);
    Eigen::VectorXd b(nADJ+1);
    Eigen::VectorXd C(4);

    // Assemble adjacent interpolation equations
    for (int i = 0; i < nADJ; i++)
    {
        b[i] = val_adj[i];

        dx = p_adj[i].x() - p_src.x();
        dy = p_adj[i].y() - p_src.y();
        dz = p_adj[i].z() - p_src.z();

        A(i, 0) = 1.0;
        A(i, 1) = dx;
        A(i, 2) = dy;
        A(i, 3) = dz;
    }

    // Incoporate the boundary condition
    b[nADJ] = bc_gamma;
    A(nADJ, 0) = bc_beta;
    A(nADJ, 1) = bc_alpha * n_src.x();
    A(nADJ, 2) = bc_alpha * n_src.y();
    A(nADJ, 3) = bc_alpha * n_src.z();

    // Solve least-squares coefficients
    C = A.colPivHouseholderQr().solve(b);

    // Interpolate at the image point
    const Foam::vector p_img = 2.0 * p_src - p_dst;
    dx = p_img.x() - p_src.x();
    dy = p_img.y() - p_src.y();
    dz = p_img.z() - p_src.z();
    const double val_img = C[0] + C[1] * dx + C[2] * dy + C[3] * dz;

    // Interpolate at the ghost point
    double snGrad = (bc_gamma - bc_beta * C[0]) / bc_alpha;
    double d = Foam::mag(p_dst - p_img);
    val_dst = val_img - snGrad * d;
}

/**
 * Gas-phase properties.
 */
void calc_gas_property(double T, double &lambda_, double &mu_, double &Cp_)
{
    lambda_ = 1.08e-4 * T + 0.0133;              // Unit: W/m/K
    Cp_ = 0.3 * kcal2J;                          // Unit: J/kg/K
    mu_ = Pr * lambda_ / Cp_;                    // Unit: kg/m/s
}

/* Main program */
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"

    /* Load meshes */
    #include "createMeshes.H"

    /* Create variables */
    #include "createFields.H"

    /* Initialize the gas-phase */
    {
        // Density
        rho = p / Rg / T;
        rho.correctBoundaryConditions();

        // Signed-distance
        for (int i = 0; i < pointMesh_gas.size(); i++)
        {
            phi_gas[i] = mesh_gas.points()[i].z() - plane_z;
        }

        // Properties
        for (int i = 0; i < mesh_gas.nCells(); i++)
        {
            calc_gas_property(T[i], lambda[i], mu[i], Cp[i]);
        }
        mu.correctBoundaryConditions();
        Cp.correctBoundaryConditions();
        lambda.correctBoundaryConditions();
    }

    /* Initialize the solid-phase */
    {
        for (int i = 0; i < mesh_solid.nCells(); i++)
        {
            T_solid[i] = T0;
            if (mesh_solid.C()[i].x() < 0.3*L || mesh_solid.C()[i].x() > 0.7*L)
            {
                rho_solid[i] = rho_AP;
                c[i] = c_AP;
                lambda_solid[i] = lambda_AP;
            }
            else
            {
                rho_solid[i] = rho_HTPB;
                c[i] = c_HTPB;
                lambda_solid[i] = lambda_HTPB;
            }
        }
        T_solid.correctBoundaryConditions();
        rho_solid.correctBoundaryConditions();
        c.correctBoundaryConditions();
        lambda_solid.correctBoundaryConditions(); 

        for (int i = 0; i < pointMesh_solid.size(); i++)
            phi_solid[i] = pointMesh_solid.mesh().points()[i].z() - 0.95*L;
        phi_solid.correctBoundaryConditions();
    }

    /* createPhi.H */
    USn = Foam::fvc::interpolate(U) & mesh_gas.Sf();
    setBdryVal(mesh_gas, U, USn);

    /* compressibleCreatePhi.H */
    rhoUSn = Foam::fvc::interpolate(rho*U) & mesh_gas.Sf();
    setBdryVal(mesh_gas, rho, U, rhoUSn);

    /* Classify mesh cells */
    identifyIBCell(mesh_gas, phi_gas, cIbMarker);
    for (int i = 0; i < mesh_gas.nCells(); i++)
        cIbMask[i] = cIbMarker[i] > 0 ? 1.0 : 0.0;

    /* Extended stencil */
    const Foam::extendedCentredCellToCellStencil& addressing = Foam::centredCPCCellToCellStencilObject::New(mesh_gas); // Processor-boundaries are dealt with automatically
    // Foam::Pout << addressing.stencil().size() << "/" << mesh_gas.nCells() << Foam::endl;

    // Initialize stencil storage
    Foam::List<Foam::vectorList> sten_pos(mesh_gas.nCells());
    Foam::List<Foam::scalarList> sten_ib(mesh_gas.nCells());
    addressing.collectData(mesh_gas.C(), sten_pos);
    addressing.collectData(cIbMarker, sten_ib);

    Foam::List<Foam::vectorList> pntAdjCentroid;
    collectData_pointCell(pointMesh_solid, flag_pntOnParBdry_solid, mesh_solid.C(), pntAdjCentroid);

    while(runTime.loop())
    {
        const Foam::dimensionedScalar dt(Foam::dimTime, runTime.deltaTValue());
        Foam::Info << "\nn=" << runTime.timeIndex() << ", t=" << std::stod(runTime.timeName(), nullptr)*s2ms << "ms, dt=" << dt.value()*s2ns << "ns" << Foam::endl;
        runTime.write();

        /* Interpolation on IB cells */
        {
            // Collect data
            Foam::List<Foam::vectorList> sten_val_U(mesh_gas.nCells());
            Foam::List<Foam::scalarList> sten_val_p(mesh_gas.nCells());
            Foam::List<Foam::scalarList> sten_val_T(mesh_gas.nCells());
            addressing.collectData(U, sten_val_U);
            addressing.collectData(p, sten_val_p);
            addressing.collectData(T, sten_val_T);

            // Interpolate data
            for (int i = 0; i < mesh_gas.nCells(); i++)
            {
                if (isGhostCell(cIbMarker[i]))
                {
                    // Position and normal of the Boundary Intersection point
                    const Foam::vector p_BI(mesh_gas.C()[i].x(), mesh_gas.C()[i].y(), plane_z);
                    const Foam::vector n_BI(0.0, 0.0, 1.0);

                    // Extract neighborhood data
                    std::vector<Foam::label> idx;
                    for (int j = 0; j < addressing.stencil()[i].size(); j++)
                    {
                        if (isCellCentroid(sten_pos[i][j]) && isEqual(sten_ib[i][j], cFluid))
                            idx.push_back(j);
                    }
                    std::vector<Foam::vector> pos(idx.size());
                    std::vector<Foam::scalar> val_ux(idx.size()), val_uy(idx.size()), val_uz(idx.size());
                    std::vector<Foam::scalar> val_p(idx.size());
                    std::vector<Foam::scalar> val_T(idx.size());
                    for (int j = 0; j < idx.size(); j++)
                    {
                        const auto jloc = idx[j];
                        pos[j] = sten_pos[i][jloc];
                        val_ux[j] = sten_val_U[i][jloc].x();
                        val_uy[j] = sten_val_U[i][jloc].y();
                        val_uz[j] = sten_val_U[i][jloc].z();
                        val_p[j] = sten_val_p[i][jloc];
                        val_T[j] = sten_val_T[i][jloc];
                    }

                    // Velocity
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_ux, mesh_gas.C()[i], U[i].x());
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_uy, mesh_gas.C()[i], U[i].y());
                    ibInterp_Dirichlet_Linear(p_BI, 1.0, pos, val_uz, mesh_gas.C()[i], U[i].z());

                    // Pressure
                    ibInterp_zeroGradient_Linear(p_BI, n_BI, pos, val_p, mesh_gas.C()[i], p[i]);

                    // Temperature
                    ibInterp_Dirichlet_Linear(p_BI, plane_T, pos, val_T, mesh_gas.C()[i], T[i]);
                }
            }
        }

        /* Update gas-phase by semi-implicit iteration */
        bool converged = false;
        int m = 0;
        while(++m <= 5)
        {
            Foam::Info << "m=" << m << Foam::endl;

            /* Predictor */
            {
                /* Provisional density */
                {
                    // Discretized equation
                    Foam::fvScalarMatrix rhoEqn
                    (
                        Foam::fvm::ddt(rho) + Foam::fvc::div(rhoUSn)
                    );

                    // Solve
                    rhoEqn.solve();

                    // Record for further use
                    rho_star = rho;
                }

                /* Temperature */
                {
                    // Discretized equation
                    Foam::fvScalarMatrix TEqn
                    (
                        Foam::fvm::ddt(rho, T) + Foam::fvc::div(rhoUSn, T) == Foam::fvc::laplacian(lambda, T) / Cp
                    );

                    // For solid cells, set temperature to constant;
                    // For ghost cells, set to interpolated value.
                    {
                        Foam::scalarList val;
                        Foam::labelList idx;

                        for (int i=0; i < mesh_gas.nCells(); i++)
                        {
                            if (cIbMarker[i] < cFluid)
                            {
                                idx.append(i);
                                if (isEqual(cIbMarker[i], cSolid))
                                    val.append(T0);
                                else
                                    val.append(T[i]);
                            }
                        }
                        TEqn.setValues(idx, val);
                    }

                    // Solve
                    TEqn.solve();
                }

                /* Provisional momentum */
                {
                    grad_p = Foam::fvc::grad(p);

                    // Discretized equation
                    Foam::fvVectorMatrix UEqn
                    (
                        Foam::fvm::ddt(rho, U) + Foam::fvc::div(rhoUSn, U) == -grad_p + Foam::fvc::laplacian(mu, U)
                    );

                    // For solid cells, set velocity to zero;
                    // For ghost cells, set to the interpolated value.
                    {
                        Foam::vectorList val;
                        Foam::labelList idx;

                        for (int i=0; i < mesh_gas.nCells(); i++)
                        {
                            if (cIbMarker[i] < cFluid)
                            {
                                idx.append(i);
                                if (isEqual(cIbMarker[i], cSolid))
                                    val.append(Foam::vector(0.0, 0.0, 0.0));
                                else
                                    val.append(U[i]);
                            }
                        }
                        UEqn.setValues(idx, val);
                    }

                    // Solve
                    UEqn.solve();
                }

                /* Provisional mass flux */
                rhoUSn = Foam::fvc::interpolate(rho_star*U) & mesh_gas.Sf();
                Foam::surfaceScalarField grad_p_f_compact(Foam::fvc::snGrad(p) * mesh_gas.magSf());
                Foam::surfaceScalarField grad_p_f_mean(Foam::fvc::interpolate(grad_p) & mesh_gas.Sf());
                rhoUSn -= dt * (grad_p_f_compact - grad_p_f_mean);
                setBdryVal(mesh_gas, rho, U, rhoUSn);

                /* Density (thermal) */
                thermo.correct();
                rho = p / Rg / T;
                rho.correctBoundaryConditions();
            }

            /* Corrector */
            {
                /* Pressure-correction Helmholtz equation */
                // Discretized equation
                Foam::fvScalarMatrix dpEqn
                (
                    Foam::fvm::Sp(1.0/(dt * Rg * T), dp) - dt * Foam::fvm::laplacian(dp) == -(Foam::fvc::div(rhoUSn) + Foam::fvc::ddt(rho))
                );
                // For both solid and ghost cells, set the incremental pressure-correction to zero.
                {
                    Foam::scalarList val;
                    Foam::labelList idx;
                    for (int i=0; i < mesh_gas.nCells(); i++)
                    {
                        if (cIbMarker[i] < cFluid)
                        {
                            idx.append(i);
                            val.append(0.0);
                        }
                    }
                    dpEqn.setValues(idx, val);
                }
                // Solve
                dpEqn.solve();

                /* Update */
                p += dp;
                p.correctBoundaryConditions();

                U = (rho_star*U - dt * Foam::fvc::grad(dp)) / rho;
                U.correctBoundaryConditions();

                rhoUSn -= dt * Foam::fvc::snGrad(dp) * mesh_gas.magSf();
                setBdryVal(mesh_gas, rho, U, rhoUSn);

                USn = rhoUSn / Foam::fvc::interpolate(rho);
                setBdryVal(mesh_gas, U, USn);
            }

            /* Check convergence */
            {
                Foam::scalar eps_inf, eps_1, eps_2;

                diagnose(mesh_gas, dp, cIbMask, eps_1, eps_2, eps_inf);
                Foam::Info << "||dp||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                const bool criteria_dp = eps_inf < 1e-3 * p0 && eps_1 < 1e-6 * p0 && eps_2 < 1e-4 * p0;

                diagnose(mesh_gas, rho-rho_star, cIbMask, eps_1, eps_2, eps_inf);
                Foam::Info << "||rho(m)-rho*||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                const bool criteria_drho = eps_inf < 1e-3 || eps_1 < 1e-6 || eps_2 < 1e-5;

                converged = criteria_dp && criteria_drho;
                if (converged && m > 1)
                    break;
            }
        }
        if (!converged)
            Foam::Perr << "\nGas-phase failed to converged after " << m-1 << " semi-implicit iterations!" << Foam::endl;

        /* Update gas-phase properties */
        for (int i = 0; i < mesh_gas.nCells(); i++)
        {
            calc_gas_property(T[i], lambda[i], mu[i], Cp[i]);
        }
        mu.correctBoundaryConditions();
        Cp.correctBoundaryConditions();
        lambda.correctBoundaryConditions();

        /* Check range */
        {
            Foam::scalar vMin, vMax;

            // Density
            diagnose(mesh_gas, rho, cIbMask, vMin, vMax);
            Foam::Info << "\nrho: " << vMin << " ~ " << vMax << Foam::endl;

            // Velocity magnitude
            diagnose(mesh_gas, Foam::mag(U), cIbMask, vMin, vMax);
            Foam::Info << "|U|: " << vMin << " ~ " << vMax << Foam::endl;

            // Pressure fluctuation
            diagnose(mesh_gas, p, cIbMask, vMin, vMax);
            Foam::Info << "p-p0: " << vMin-p0 << " ~ " << vMax-p0 << Foam::endl;

            // Temperature
            diagnose(mesh_gas, T, cIbMask, vMin, vMax);
            Foam::Info << "T: " << vMin << " ~ " << vMax << Foam::endl;
        }

        /* Update the position of gas-solid interface */
        {
            // Extension velocity, Unit: m/s
            for (int i = 0; i < pointMesh_solid.size(); i++)
                F[i] = 2 * mm2m;

            // Gradient of the Level-Set fucntion (pointScalarField -> volVectorField)
            for (int i = 0; i < mesh_solid.nCells(); i++)
            {
                Foam::vector g(0.0, 0.0, 0.0);
                const auto &fL = mesh_solid.cells()[i];
                for(int j = 0; j < fL.size(); j++)
                {
                    const auto fI = fL[j];
                    const auto &f = mesh_solid.faces()[fI];

                    Foam::scalar val = 0.0;
                    for(int k = 0; k < f.size(); k++)
                    {
                        const auto pI = f[k];
                        val += phi_solid[pI];
                    }
                    val /= f.size();

                    const Foam::vector &Sf = mesh_solid.Sf()[fI];
                    const Foam::vector d_Cf = mesh_solid.Cf()[fI] - mesh_solid.C()[i];
                    const Foam::vector Sn = (d_Cf & Sf) < 0.0 ? -Sf : Sf;

                    g += val * Sn;
                }
                grad_phi_solid[i] = g / mesh_solid.V()[i];
            }

            // Collect data on surrounding cells for each point
            Foam::List<Foam::vectorList> p2C_gradPhi;
            collectData_pointCell(pointMesh_solid, flag_pntOnParBdry_solid, grad_phi_solid, p2C_gradPhi);

            // Upwind reconstruction of the gradient on points (centroids -> points)
            for (int i = 0; i < pointMesh_solid.size(); i++)
            {
                const Foam::vector &P = mesh_solid.points()[i];
                const Foam::vectorList &gradPhi = p2C_gradPhi[i];
                const Foam::vectorList &adjC = pntAdjCentroid[i];
                Foam::vector &gradPhi_dst = grad_phi_upwind_solid[i];

                // Check size
                if (adjC.size() != gradPhi.size())
                {
                    Foam::Perr << "Inconsistant number of surrounding cells of point[" << i << "]: " << adjC.size() << "/" << gradPhi.size() << Foam::endl;
                    return 110;
                }
                const int nAdjCell = adjC.size();
                if (nAdjCell < mesh_solid.pointCells()[i].size() || nAdjCell == 0)
                {
                    Foam::Perr << "point[" << i << "]: nAdjCell=" << nAdjCell
                               << ", pointCells().size()=" << mesh_solid.pointCells()[i].size()
                               << ", flag_pntOnBdry=" << flag_pntOnBdry_solid[i]
                               << ", flag_pntOnPhyBdry=" << flag_pntOnPhyBdry_solid[i]
                               << ", flag_pntOnParBdry=" << flag_pntOnParBdry_solid[i]
                               << Foam::endl;
                }

                // Approximate normal direction
                Foam::vector n(0.0, 0.0, 0.0);
                Foam::scalar w_sum = 0.0;
                for (int j = 0; j < nAdjCell; j++)
                {
                    const Foam::vector &C = adjC[j];
                    const Foam::vector A = P - C;
                    const Foam::scalar d = Foam::mag(A);
                    const Foam::scalar w = 1.0 / d;
                    w_sum += w;
                    n += w * gradPhi[j];
                }
                n /= w_sum;

                // Calculate the upwind weight
                gradPhi_dst.x() = 0.0;
                gradPhi_dst.y() = 0.0;
                gradPhi_dst.z() = 0.0;
                Foam::scalar gamma_sum = 0.0;
                for (int j = 0; j < nAdjCell; j++)
                {
                    const Foam::vector &C = adjC[j];
                    const Foam::vector A = P - C;
                    const Foam::scalar gamma = std::max(0.0, n&A/Foam::mag(A));
                    gamma_sum += gamma;
                    gradPhi_dst += gamma * gradPhi[j];
                }
                if (isZero(gamma_sum))
                {
                    gamma_sum = 0.0;
                    gradPhi_dst *= gamma_sum;
                    for (int j = 0; j < nAdjCell; j++)
                    {
                        const Foam::vector &C = adjC[j];
                        const Foam::vector A = P - C;
                        const Foam::scalar gamma = 1.0 / Foam::mag(A);
                        gamma_sum += gamma;
                        gradPhi_dst += gamma * gradPhi[j];
                    }
                }
                gradPhi_dst /= gamma_sum;
            }

            // Upwind evaluation of the Hamiltonian
            for (int i = 0; i < pointMesh_solid.size(); i++)
            {
                const Foam::scalar H = F[i] * Foam::mag(grad_phi_upwind_solid[i]);
                phi_solid[i] -= dt.value() * H;
            }
        }
    }

    return 0;
}
