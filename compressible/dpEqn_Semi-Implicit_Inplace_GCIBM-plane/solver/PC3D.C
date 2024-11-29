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

/* Classification of gas-phase cells */
const Foam::scalar cFluid = 1.0;                 // Fluid cell
const Foam::scalar cGhost = 0.0;                 // Ghost cell
const Foam::scalar cSolid = -1.0;                // Solid cell

/* Flow condition */
const Foam::scalar Re = 40.0;                    // Reynolds number
const Foam::scalar Pr = 0.71;                    // Prandtl number
const Foam::scalar p0 = 101325.0;                // Ambient pressure
const Foam::scalar T0 = 300.0;                   // Initial temperature

/* Plane param */
const Foam::scalar plane_z = 0.25e-3;            // Vertical position
const Foam::scalar plane_T = 300.0;              // Temperature

/* Cartesian grid */
const Foam::scalar xMin = 0.0, xMax = 0.5e-3;    // Range in X-direction
const Foam::scalar yMin = 0.0, yMax = 0.5e-3;    // Range in Y-direction
const Foam::scalar zMin = 0.0, zMax = 1.0e-3;    // Range in Z-direction
const Foam::scalar h = 0.5e-3 / 32;              // Spacing
const Foam::scalar h_inv = 1.0 / h;

inline bool isEqual(double x, double y)
{
    return std::islessequal(std::abs(x-y), 1e-5 * std::abs(x));
}

inline bool isZero(double x)
{
    return std::isless(std::abs(x), 1e-7);
}

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

inline bool isCellCentroid(const Foam::vector &p)
{
    return atHalfSpacing(p.x(), xMin, h) && atHalfSpacing(p.y(), yMin, h) && atHalfSpacing(p.z(), zMin, h);
}

inline bool isNode(const Foam::vector &p)
{
    return atFullSpacing(p.x(), xMin, h) && atFullSpacing(p.y(), yMin, h) && atFullSpacing(p.z(), zMin, h);
}

inline void xyz2ijk(const Foam::vector &p_, int &i_, int &j_, int &k_)
{
    const double dx = p_.x() - xMin;
    const double dy = p_.y() - yMin;
    const double dz = p_.z() - zMin;
    i_ = static_cast<int>(std::fma(dx, h_inv, 0.5));
    j_ = static_cast<int>(std::fma(dy, h_inv, 0.5));
    k_ = static_cast<int>(std::fma(dz, h_inv, 0.5));
}

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

void diagnose(const Foam::fvMesh &mesh, const Foam::globalMeshData &meshInfo, const Foam::volScalarField &src, const Foam::volScalarField &flag, double &norm1, double &norm2, double &normInf, double &minVal, double &maxVal)
{
    norm1 = norm2 = normInf = 0.0;
    minVal = std::numeric_limits<double>::max();
    maxVal = std::numeric_limits<double>::min();
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isZero(flag[i]))
            continue;

        const double cVal = std::abs(src[i]);
        normInf = std::max(normInf, cVal);
        norm1 += cVal;
        norm2 += cVal*cVal;
        minVal = std::min(minVal, src[i]);
        maxVal = std::max(maxVal, src[i]);
    }
    Foam::reduce(normInf, Foam::maxOp<Foam::scalar>());
    Foam::reduce(norm1, Foam::sumOp<Foam::scalar>());
    Foam::reduce(norm2, Foam::sumOp<Foam::scalar>());
    norm1 /= meshInfo.nTotalCells();
    norm2 = std::sqrt(norm2/meshInfo.nTotalCells());
    Foam::reduce(minVal, Foam::minOp<Foam::scalar>());
    Foam::reduce(maxVal, Foam::maxOp<Foam::scalar>());
}

void diagnose(const Foam::fvMesh &mesh, const Foam::globalMeshData &meshInfo, const Foam::volScalarField &src, const Foam::volScalarField &flag, double &norm1, double &norm2, double &normInf)
{
    norm1 = norm2 = normInf = 0.0;
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isZero(flag[i]))
            continue;

        const double cVal = std::abs(src[i]);
        normInf = std::max(normInf, cVal);
        norm1 += cVal;
        norm2 += cVal*cVal;
    }
    Foam::reduce(normInf, Foam::maxOp<Foam::scalar>());
    Foam::reduce(norm1, Foam::sumOp<Foam::scalar>());
    Foam::reduce(norm2, Foam::sumOp<Foam::scalar>());
    norm1 /= meshInfo.nTotalCells();
    norm2 = std::sqrt(norm2/meshInfo.nTotalCells());
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

inline double distToSurf(const Foam::vector &c)
{
    return c.z() - plane_z;
}

inline bool pntInFluid(const Foam::vector &c)
{
    return distToSurf(c) > 0.0;
}

inline bool pntInSolid(const Foam::vector &c)
{
    return !pntInFluid(c);
}

inline bool pntNearSurf(const Foam::vector &c)
{
    return std::abs(distToSurf(c)) < 0.1 * h;
}

bool cellIsIntersected(const Foam::vectorList &v)
{
    bool flag = pntInSolid(v[0]);
    for (int i = 1; i < v.size(); i++)
    {
        if (pntInSolid(v[i]) != flag)
            return true;
    }
    return false;
}

Foam::scalar cellNum(const Foam::vectorList &v, const Foam::vector &c)
{
    if (pntInFluid(c))
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
                if (pntNearSurf(v[i]))
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

void identifyIBCell(const Foam::fvMesh &mesh, Foam::volScalarField &marker)
{
    int nSolid = 0, nGhost = 0, nFluid = 0;
    for (int i = 0; i < mesh.nCells(); i++)
    {
        const auto &c = mesh.C()[i];

        const auto &cP = mesh.cellPoints()[i];
        Foam::vectorList p(cP.size());
        for (int j = 0; j < p.size(); j++)
            p[j] = mesh.points()[cP[j]];

        marker[i] = cellNum(p, c);

        if (isEqual(marker[i], cSolid))
            ++nSolid;
        else if (isEqual(marker[i], cFluid))
            ++nFluid;
        else
            ++nGhost;
    }

    // Ensure all solid cells not adjacent to fluid cells directly (through face) 
    for (int i = 0; i < mesh.nCells(); i++)
    {
        if (isEqual(marker[i], cSolid))
        {
            const auto &cC = mesh.cellCells()[i];
            for (int j = 0; j < cC.size(); j++)
            {
                const auto adjCI = cC[j];
                if (isEqual(marker[adjCI], cFluid))
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
            if (isEqual(marker[i], cGhost))
            {
                const auto &cC = mesh.cellCells()[i];
                bool adjToSolid = false;
                for (int j = 0; j < cC.size(); j++)
                {
                    const auto adjCI = cC[j];
                    if (isEqual(marker[adjCI], cSolid))
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
            if (isEqual(marker[i], cGhost))
            {
                const auto &cC = mesh.cellCells()[i];
                bool adjToFluid = false;
                for (int j = 0; j < cC.size(); j++)
                {
                    const auto adjCI = cC[j];
                    if (isEqual(marker[adjCI], cFluid))
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

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"

    /* Load meshes */
    #include "createMeshes.H"

    /* Create variables */
    #include "createFields.H"

    /* Initialize */
    {
        // Density
        rho = p / Rg / T;
        rho.correctBoundaryConditions();

        // Signed-distance
        for (int i = 0; i < mesh_gas.nCells(); i++)
        {
            phi[i] = distToSurf(mesh_gas.C()[i]);
        }

        // Properties
        for (int i = 0; i < mesh_gas.nCells(); i++)
        {
            mu[i] = rho[i] / Re;                 // Unit: kg/m/s. (D=1m, U=1m/s by default)
            Cp[i] = 0.3 * kcal2J;                // Unit: J/kg/K
            lambda[i] = mu[i] * Cp[i] / Pr;      // Unit: W/m/K
        }
        mu.correctBoundaryConditions();
        Cp.correctBoundaryConditions();
        lambda.correctBoundaryConditions();
    }

    /* createPhi.H */
    USn = Foam::fvc::interpolate(U) & mesh_gas.Sf();
    setBdryVal(mesh_gas, U, USn);

    /* compressibleCreatePhi.H */
    rhoUSn = Foam::fvc::interpolate(rho*U) & mesh_gas.Sf();
    setBdryVal(mesh_gas, rho, U, rhoUSn);

    /* Classify mesh cells */
    identifyIBCell(mesh_gas, cIbMarker);
    for (int i = 0; i < mesh_gas.nCells(); i++)
        cIbMask[i] = cIbMarker[i] > 0 ? 1.0 : 0.0;

    /* Extended stencil */
    const Foam::extendedCentredCellToCellStencil& addressing = Foam::centredCPCCellToCellStencilObject::New(mesh_gas); // Processor-boundaries are dealt with automatically
    Foam::Pout << addressing.stencil().size() << "/" << mesh_gas.nCells() << Foam::endl;

    // Initialize stencil storage
    Foam::List<Foam::vectorList> sten_pos(mesh_gas.nCells());
    Foam::List<Foam::scalarList> sten_ib(mesh_gas.nCells());
    addressing.collectData(mesh_gas.C(), sten_pos);
    addressing.collectData(cIbMarker, sten_ib);

    while(runTime.loop())
    {
        const Foam::dimensionedScalar dt(Foam::dimTime, runTime.deltaTValue());
        Foam::Info << "\nn=" << runTime.timeIndex() << ", t=" << std::stod(runTime.timeName(), nullptr) << "s, dt=" << dt.value()*s2ms << "ms" << Foam::endl;
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
                if (isEqual(cIbMarker[i], cGhost))
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
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_uz, mesh_gas.C()[i], U[i].z());

                    // Pressure
                    ibInterp_zeroGradient_Linear(p_BI, n_BI, pos, val_p, mesh_gas.C()[i], p[i]);

                    // Temperature
                    ibInterp_Dirichlet_Linear(p_BI, plane_T, pos, val_T, mesh_gas.C()[i], T[i]);
                    // ibInterp_zeroGradient_Linear(p_BI, n_BI, pos, val_T, mesh_gas.C()[i], T[i]);
                    // ibInterp_Neumann_Linear(p_BI, n_BI, -2500.0, pos, val_T, mesh_gas.C()[i], T[i]);
                    // ibInterp_Robin_Linear(p_BI, n_BI, 2.0, 10.0, 8000.0, pos, val_T, mesh_gas.C()[i], T[i]);
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
                    // For ghost cells, set temperature to interpolated value.
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
                    // For ghost cells, set velocity to interpolated value.
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
                rho = p / Rg / T;
                rho.correctBoundaryConditions();
            }

            /* Corrector */
            {
                /* Pressure-correction Helmholtz equation */
                {
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
                }

                /* Update */
                {
                    p += dp;
                    p.correctBoundaryConditions();

                    U = (rho_star*U - dt * Foam::fvc::grad(dp)) / rho;
                    U.correctBoundaryConditions();

                    rhoUSn -= dt * Foam::fvc::snGrad(dp) * mesh_gas.magSf();
                    setBdryVal(mesh_gas, rho, U, rhoUSn);

                    USn = rhoUSn / Foam::fvc::interpolate(rho);
                    setBdryVal(mesh_gas, U, USn);
                }
            }

            /* Check convergence */
            {
                Foam::scalar eps_inf, eps_1, eps_2;

                diagnose(mesh_gas, meshInfo_gas, dp, cIbMask, eps_1, eps_2, eps_inf);
                Foam::Info << "||dp||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                const bool criteria_dp = eps_inf < 1e-2;

                diagnose(mesh_gas, meshInfo_gas, rho-rho_star, cIbMask, eps_1, eps_2, eps_inf);
                Foam::Info << "||rho_next-rho*||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                const bool criteria_drho = eps_inf < 1e-3 || eps_1 < 1e-5 || eps_2 < 1e-6;

                converged = criteria_dp || criteria_drho;
                if (converged && m > 1)
                    break;
            }
        }
        if (!converged)
            Foam::Info << Foam::nl  << "Gas-phase failed to converged after " << m << " semi-implicit iterations!" << Foam::endl;

        /* Check range */
        {
            Foam::scalar eps_inf, eps_1, eps_2;
            Foam::scalar vMin, vMax;

            // Density
            diagnose(mesh_gas, rho, cIbMask, vMin, vMax);
            Foam::Info << Foam::nl;
            Foam::Info << "rho: " << vMin << " ~ " << vMax << Foam::endl;

            // Velocity magnitude
            diagnose(mesh_gas, Foam::mag(U), cIbMask, vMin, vMax);
            Foam::Info << "|U|: " << vMin << " ~ " << vMax << Foam::endl;

            // Pressure fluctuation
            diagnose(mesh_gas, p, cIbMask, vMin, vMax);
            Foam::Info << "p-p0: " << vMin-p0 << " ~ " << vMax-p0 << Foam::endl;

            // Temperature
            diagnose(mesh_gas, T, cIbMask, vMin, vMax);
            Foam::Info << "T: " << vMin << " ~ " << vMax << Foam::endl;

            // Continuity
            diagnose(mesh_gas, meshInfo_gas, Foam::fvc::div(rhoUSn), cIbMask, eps_1, eps_2, eps_inf);
            Foam::Info << "||div(rhoU)||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
        }

    }

    return 0;
}
