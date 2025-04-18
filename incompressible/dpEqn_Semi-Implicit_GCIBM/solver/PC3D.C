#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "fvc.H"
#include "fvm.H"
#include "timeSelector.H"
#include "meshTools.H"
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
const Foam::scalar one_atm = 101325.0;  // Unit: Pa
const Foam::scalar one_bar = 100000.0;  // Unit: Pa
const Foam::scalar one_psi = 6894.76;   // Unit: Pa
const Foam::scalar one_mpa = 1e6;       // Unit: Pa
const Foam::scalar G0 = 1.4;            // Specific heat ratio for ideal gas.

/* Classification of gas-phase cells */
const Foam::scalar cFluid = 1.0;
const Foam::scalar cGhost = 0.0;
const Foam::scalar cSolid = -1.0;

/* Flow condition */
const Foam::scalar Re = 40.0;  // Reynolds number

/* Sphere parameter */
const Foam::vector sphere_c0(0, 0, 0);
const Foam::scalar sphere_r = 0.5;

/* Cartesian grid spacing */
const Foam::scalar h = 8.0 / 64;
const Foam::scalar xMin = -4.0, xMax = 4.0;
const Foam::scalar yMin = -4.0, yMax = 4.0;
const Foam::scalar zMin = -4.0, zMax = 12.0;

inline bool isEqual(double x, double y)
{
    return std::abs(x-y) <= 1e-6 * std::abs(x);
}

inline double normalizedCoord(double dx_, double h_)
{
    return dx_ / h_;
}

inline bool atFullSpacing(double dx_, double h_)
{
    return isEqual(std::remainder(normalizedCoord(dx_, h_), 1.0), 0.0);
}

inline bool atHalfSpacing(double dx_, double h_)
{
    return isEqual(std::remainder(normalizedCoord(dx_, h_), 0.5), 0.0);
}

inline bool isCellCentroid(const Foam::vector &p)
{
    return atHalfSpacing(p.x() - xMin, h) && atHalfSpacing(p.y() - yMin, h) && atHalfSpacing(p.z() - zMin, h);
}

inline bool isNode(const Foam::vector &p)
{
    return atFullSpacing(p.x() - xMin, h) && atFullSpacing(p.y() - yMin, h) && atFullSpacing(p.z() - zMin, h);
}

void xyz2ijk(double x_, double y_, double z_, int &i_, int &j_, int &k_)
{
    i_ = static_cast<int>(normalizedCoord(x_ - xMin, h) + 0.5);
    j_ = static_cast<int>(normalizedCoord(y_ - yMin, h) + 0.5);
    k_ = static_cast<int>(normalizedCoord(z_ - zMin, h) + 0.5);
}

void update_boundaryField(const Foam::fvMesh &mesh, const Foam::volScalarField &src1, const Foam::volVectorField &src2, Foam::surfaceScalarField &dst)
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

inline double distToSurf(const Foam::vector &c)
{
    return Foam::mag(c-sphere_c0) - sphere_r;
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
    const auto d = distToSurf(c);
    return std::abs(d) < 0.1 * h;
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

    /* Load meshes and create variables */
    #include "createMeshAndField.H"

    /* compressibleCreatePhi.H */
    rhoUSn = Foam::fvc::interpolate(rhoU) & mesh_gas.Sf();
    update_boundaryField(mesh_gas, rho, U, rhoUSn);

    /* Initialize */
    {
        for (int i = 0; i < mesh_gas.nCells(); i++)
        {
            p[i] = 0.0;
            U[i].x() = 0.0;
            U[i].y() = 0.0;
            U[i].z() = 0.0;
            rho[i] = 1.0;
            rhoU[i] = rho[i] * U[i];
            mu[i] = 1.0 / Re;
        }
        p.correctBoundaryConditions();
        U.correctBoundaryConditions();
        rho.correctBoundaryConditions();
        rhoU.correctBoundaryConditions();
        mu.correctBoundaryConditions();

        U_next = U;
        p_next = p;
        rhoU_next = rhoU;

        U_next.correctBoundaryConditions();
        p_next.correctBoundaryConditions();
        rhoU_next.correctBoundaryConditions();
    }

    /* Classify mesh cells */
    identifyIBCell(mesh_gas, cMarker);
    for (int i = 0; i < mesh_gas.nCells(); i++)
    {
        if (cMarker[i] > 0)
            cIbMask[i] = 1.0;
        else
            cIbMask[i] = 0.0;
    }

    /* Extended stencil  */
    // Processor-boundaries are dealt with automatically
    const Foam::extendedCentredCellToCellStencil& addressing = Foam::centredCPCCellToCellStencilObject::New(mesh_gas);
    Foam::Pout << addressing.stencil().size() << "/" << mesh_gas.nCells() << Foam::endl;

    // Initialize stencil storage
    Foam::List<Foam::vectorList> sten_pos(mesh_gas.nCells());
    Foam::List<Foam::scalarList> sten_ib(mesh_gas.nCells());
    addressing.collectData(mesh_gas.C(), sten_pos);
    addressing.collectData(cMarker, sten_ib);

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
            addressing.collectData(U, sten_val_U);
            addressing.collectData(p, sten_val_p);

            // Interpolate data
            for (int i = 0; i < mesh_gas.nCells(); i++)
            {
                if (isEqual(cMarker[i], cGhost))
                {
                    // Position and normal of the Boundary Intersection point
                    Foam::vector p_BI = mesh_gas.C()[i] - sphere_c0;
                    Foam::vector n_BI = p_BI / Foam::mag(p_BI);
                    p_BI = sphere_c0 + n_BI * sphere_r;

                    // Extract neighborhood data
                    std::vector<Foam::label> idx;
                    for (int j = 0; j < addressing.stencil()[i].size(); j++)
                    {
                        if (isCellCentroid(sten_pos[i][j]) && isEqual(sten_ib[i][j], cFluid))
                            idx.push_back(j);
                    }
                    std::vector<Foam::vector> pos(idx.size());
                    std::vector<Foam::scalar> val_ux(idx.size()), val_uy(idx.size()), val_uz(idx.size()), val_p(idx.size());
                    for (int j = 0; j < idx.size(); j++)
                    {
                        const auto jloc = idx[j];
                        pos[j] = sten_pos[i][jloc];
                        val_ux[j] = sten_val_U[i][jloc].x();
                        val_uy[j] = sten_val_U[i][jloc].y();
                        val_uz[j] = sten_val_U[i][jloc].z();
                        val_p[j] = sten_val_p[i][jloc];
                    }

                    // Velocity
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_ux, mesh_gas.C()[i], U[i].x());
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_uy, mesh_gas.C()[i], U[i].y());
                    ibInterp_Dirichlet_Linear(p_BI, 0.0, pos, val_uz, mesh_gas.C()[i], U[i].z());

                    // Pressure
                    ibInterp_zeroGradient_Linear(p_BI, n_BI, pos, val_p, mesh_gas.C()[i], p[i]);
                }
            }
        }

        /* Update gas-phase @(n) with Tw @(n) as Dirichlet B.C. and sources terms @(n) */
        {
            U_next = U;
            p_next = p;

            /* Semi-implicit iteration */
            bool converged = false;
            int m = 0;
            while(++m <= 10)
            {
                Foam::Info << "m=" << m << Foam::endl;

                /* Predictor */
                {
                    /* Provisional momentum */
                    {
                        grad_p = Foam::fvc::grad(p_next);
                        U_star = U;
                        Foam::fvVectorMatrix UEqn
                        (
                            Foam::fvm::ddt(rho, U_star) + Foam::fvc::div(rhoUSn, U_next) == -grad_p + Foam::fvc::laplacian(mu, U_next)
                        );

                        // For solid cells, set velocity to zero;
                        // For ghost cells, set velocity to interpolated value.
                        {
                            Foam::vectorList val;
                            Foam::labelList idx;

                            for (int i=0; i < mesh_gas.nCells(); i++)
                            {
                                if (cMarker[i] < cFluid)
                                {
                                    idx.append(i);
                                    if (isEqual(cMarker[i], cSolid))
                                        val.append(Foam::vector(0.0, 0.0, 0.0));
                                    else
                                        val.append(U[i]);
                                }
                            }
                            UEqn.setValues(idx, val);
                        }

                        UEqn.solve();
                    }

                    /* Provisional mass flux */
                    rhoU_star = rho * U_star;
                    rhoUSn_star = Foam::fvc::interpolate(rhoU_star) & mesh_gas.Sf();
                    Foam::surfaceScalarField grad_p_f_compact(Foam::fvc::snGrad(p_next) * mesh_gas.magSf());
                    Foam::surfaceScalarField grad_p_f_mean(Foam::fvc::interpolate(grad_p) & mesh_gas.Sf());
                    rhoUSn_star -= dt * (grad_p_f_compact - grad_p_f_mean);
                    update_boundaryField(mesh_gas, rho, U_next, rhoUSn_star);
                }

                /* Corrector */
                {
                    /* Pressure-correction Helmholtz equation */
                    Foam::fvScalarMatrix dpEqn
                    (
                        dt * Foam::fvm::laplacian(dp) == Foam::fvc::div(rhoUSn_star)
                    );

                    // For both solid and ghost cells, set the incremental pressure-correction to zero.
                    {
                        Foam::scalarList val;
                        Foam::labelList idx;

                        for (int i=0; i < mesh_gas.nCells(); i++)
                        {
                            if (cMarker[i] < cFluid)
                            {
                                idx.append(i);
                                val.append(0.0);
                            }
                        }
                        dpEqn.setValues(idx, val);
                    }

                    dpEqn.solve();

                    /* Update */
                    p_next += dp;
                    p_next.correctBoundaryConditions();

                    rhoU_next = rhoU_star - dt * Foam::fvc::grad(dp);
                    rhoU_next.correctBoundaryConditions();

                    U_next = rhoU_next / rho;
                    U_next.correctBoundaryConditions();

                    rhoUSn = rhoUSn_star - dt * Foam::fvc::snGrad(dp) * mesh_gas.magSf();
                    update_boundaryField(mesh_gas, rho, U_next, rhoUSn);

                    div_rhoU = Foam::fvc::div(rhoUSn);
                }

                /* Check convergence */
                {
                    Foam::scalar eps_inf=0.0, eps_1=0.0, eps_2=0.0;
                    for (int i = 0; i < mesh_gas.nCells(); i++)
                    {
                        const auto cVal = std::abs(dp[i]);
                        eps_inf = std::max(eps_inf, cVal);
                        eps_1 += cVal;
                        eps_2 += cVal*cVal;
                    }
                    Foam::reduce(eps_inf, Foam::maxOp<Foam::scalar>());
                    Foam::reduce(eps_1, Foam::sumOp<Foam::scalar>());
                    Foam::reduce(eps_2, Foam::sumOp<Foam::scalar>());
                    eps_1 /= meshInfo_gas.nTotalCells();
                    eps_2 = std::sqrt(eps_2/meshInfo_gas.nTotalCells());
                    Foam::Info << "||dp||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                    const bool criteria_dp = eps_inf < 1e-3;

                    eps_inf=0.0, eps_1=0.0, eps_2=0.0;
                    for (int i = 0; i < mesh_gas.nCells(); i++)
                    {
                        const auto cVal = std::abs(div_rhoU[i]);
                        eps_inf = std::max(eps_inf, cVal);
                        eps_1 += cVal;
                        eps_2 += cVal*cVal;
                    }
                    Foam::reduce(eps_inf, Foam::maxOp<Foam::scalar>());
                    Foam::reduce(eps_1, Foam::sumOp<Foam::scalar>());
                    Foam::reduce(eps_2, Foam::sumOp<Foam::scalar>());
                    eps_1 /= meshInfo_gas.nTotalCells();
                    eps_2 = std::sqrt(eps_2/meshInfo_gas.nTotalCells());
                    Foam::Info << "||div(U)||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
                    const bool criteria_div = eps_inf < 1e-3;

                    converged = criteria_dp || criteria_div;
                    if (converged && m > 1)
                        break;
                }
            }
            if (!converged)
                Foam::Info << "Gas-phase failed to converged after semi-implicit iterations!" << Foam::endl;

            /* Update to next time-level */
            U = U_next;
            p = p_next;
            rhoU = rhoU_next;

            U.correctBoundaryConditions();
            p.correctBoundaryConditions();
            rhoU.correctBoundaryConditions();
            
            /* Check range */
            {
                Foam::Info << Foam::endl;
                Foam::Info << "|U|: " << Foam::min(Foam::mag(U_next)).value() << " ~ " << Foam::max(Foam::mag(U_next)).value() << Foam::endl;
                Foam::Info << "p: " << Foam::min(p_next).value() << " ~ " << Foam::max(p_next).value() << Foam::endl;
            }
        }
    }

    return 0;
}
