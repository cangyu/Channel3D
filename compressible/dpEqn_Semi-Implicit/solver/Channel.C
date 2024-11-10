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

const Foam::dimensionedScalar R(Foam::dimEnergy/Foam::dimMoles/Foam::dimTemperature, 8.31446261815324); // Universial gas constant, Unit: J/mol/K
const Foam::dimensionedScalar MW(Foam::dimMass/Foam::dimMoles, 26e-3);                                  // Molecular weight,        Unit: kg/mol
const Foam::dimensionedScalar Rg(R/MW);

const Foam::scalar Pr = 1.0;    // Prandtl number
const Foam::scalar Le = 1.0;    // Lewis number
const Foam::scalar Re = 100.0;  // Reynolds number

const Foam::scalar p0 = 101325.0;
const Foam::scalar T0 = 300.0;

inline bool isEqual(double x, double y)
{
  return std::abs(x-y) <= 1e-6 * std::abs(x);
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

void diagnose(const Foam::fvMesh &mesh, const Foam::globalMeshData &meshInfo, const Foam::volScalarField &src, double &norm1, double &norm2, double &normInf, double &minVal, double &maxVal)
{
	norm1 = norm2 = normInf = 0.0;
	minVal = maxVal = src[0];
	for (int i = 0; i < mesh.nCells(); i++)
	{
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

void diagnose(const Foam::fvMesh &mesh, const Foam::globalMeshData &meshInfo, const Foam::volScalarField &src, double &norm1, double &norm2, double &normInf)
{
	norm1 = norm2 = normInf = 0.0;
	for (int i = 0; i < mesh.nCells(); i++)
	{
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

void diagnose(const Foam::fvMesh &mesh, const Foam::volScalarField &src, double &minVal, double &maxVal)
{
	minVal = maxVal = src[0];
	for (int i = 1; i < mesh.nCells(); i++)
	{
		minVal = std::min(minVal, src[i]);
		maxVal = std::max(maxVal, src[i]);
	}
	Foam::reduce(minVal, Foam::minOp<Foam::scalar>());
	Foam::reduce(maxVal, Foam::maxOp<Foam::scalar>());
}

int main(int argc, char *argv[])
{
	#include "setRootCase.H"
	#include "createTime.H"

	/* Load meshes and create variables */
	#include "createMeshAndField.H"

	/* Initialize */
	{
		for (int i = 0; i < mesh_gas.nCells(); i++)
		{
			p[i] = p0;
			T[i] = T0;
			U[i].x() = 0.0;
			U[i].y() = 0.0;
			U[i].z() = 0.0;
			rho[i] = p[i]/Rg.value()/T[i];

			mu[i] = rho[i] / Re; // kg/m/s, By default L=1m, U=1m/s
			Cp[i] = 0.3 * kcal2J; // J/kg/K
			lambda[i] = mu[i] * Cp[i] / Pr; // W/m/K
		}
		rho.correctBoundaryConditions();
		U.correctBoundaryConditions();
		p.correctBoundaryConditions();
		T.correctBoundaryConditions();
		mu.correctBoundaryConditions();
		lambda.correctBoundaryConditions();
		Cp.correctBoundaryConditions();
	}

	/* createPhi.H */
	USn = Foam::fvc::interpolate(U) & mesh_gas.Sf();
	setBdryVal(mesh_gas, U, USn);

	/* compressibleCreatePhi.H */
	rhoUSn = Foam::fvc::interpolate(rho*U) & mesh_gas.Sf();
	setBdryVal(mesh_gas, rho, U, rhoUSn);

	while(runTime.loop())
	{
		const Foam::dimensionedScalar dt(Foam::dimTime, runTime.deltaTValue());
		Foam::Info << "\nn=" << runTime.timeIndex() << ", t=" << std::stod(runTime.timeName(), nullptr)*s2ms << "ms, dt=" << dt.value()*s2ns << "ns" << Foam::endl;
		runTime.write();

		/* Update gas-phase @(n) with Tw @(n) as Dirichlet B.C. and sources terms @(n) */
		{
			Foam::scalar eps_inf, eps_1, eps_2;
			Foam::scalar vMin, vMax;

			/* Semi-implicit iteration */
			bool converged = false;
			int m = 0;
			while(++m < 5)
			{
				Foam::Info << "m=" << m << Foam::endl;

				/* Predictor */
				{
					/* Provisional density */
					{
						Foam::fvScalarMatrix rhoEqn
						(
							Foam::fvm::ddt(rho) + Foam::fvc::div(rhoUSn)
						);
						rhoEqn.solve();
						rho_star = rho;
					}

					/* Temperature */
					{
						Foam::fvScalarMatrix TEqn
						(
							Foam::fvm::ddt(rho, T) + Foam::fvc::div(rhoUSn, T) == Foam::fvc::laplacian(lambda, T) / Cp
						);
						TEqn.solve();
					}

					/* Provisional momentum */
					{
						grad_p = Foam::fvc::grad(p);
						Foam::fvVectorMatrix UEqn
						(
							Foam::fvm::ddt(rho, U) + Foam::fvc::div(rhoUSn, U) == -grad_p + Foam::fvc::laplacian(mu, U)
						);
						UEqn.solve();
					}

					/* Provisional mass flux */
					rhoUSn = Foam::fvc::interpolate(rho*U) & mesh_gas.Sf();
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
					Foam::fvScalarMatrix dpEqn
					(
						Foam::fvm::Sp(1.0/(dt * Rg * T), dp) - dt * Foam::fvm::laplacian(dp) == -(Foam::fvc::div(rhoUSn) + Foam::fvc::ddt(rho))
					);
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
					diagnose(mesh_gas, meshInfo_gas, dp, eps_1, eps_2, eps_inf);
					Foam::Info << "||dp||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
					const bool criteria_dp = eps_inf < 1e-3;

					diagnose(mesh_gas, meshInfo_gas, rho-rho_star, eps_1, eps_2, eps_inf);
					Foam::Info << "||rho_next-rho*||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
					const bool criteria_drho = eps_inf < 1e-3 || eps_1 < 1e-5 || eps_2 < 1e-6;

					converged = criteria_dp || criteria_drho;
					if (converged && m > 1)
						break;
				}
			}
			Foam::Info << Foam::endl;
			if (!converged)
				Foam::Info << "Gas-phase failed to converged after semi-implicit iterations!" << Foam::endl;
			
			/* Check range */
			{
				diagnose(mesh_gas, rho, vMin, vMax);
				Foam::Info << "rho: " << vMin << " ~ " << vMax << Foam::endl;

				diagnose(mesh_gas, Foam::mag(U), vMin, vMax);
				Foam::Info << "|U|: " << vMin << " ~ " << vMax << Foam::endl;
				
				diagnose(mesh_gas, p, vMin, vMax);
				Foam::Info << "p-p0: " << vMin-p0 << " ~ " << vMax-p0 << Foam::endl;

				diagnose(mesh_gas, T, vMin, vMax);
				Foam::Info << "T: " << vMin << " ~ " << vMax << Foam::endl;

				diagnose(mesh_gas, meshInfo_gas, Foam::fvc::div(rhoUSn), eps_1, eps_2, eps_inf);
				Foam::Info << "||div(rhoU)||: " << eps_inf << "(Inf), " << eps_1 << "(1), " << eps_2 << "(2)" << Foam::endl;
			}
		}
	}

    return 0;
}
