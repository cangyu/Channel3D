#include "fvCFD.H"
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

const Foam::scalar Pr = 1.0;    // Prandtl number
const Foam::scalar Le = 1.0;    // Lewis number
const Foam::scalar Re = 100.0;  // Reynolds number

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

int main(int argc, char *argv[])
{
	#include "setRootCase.H"
	#include "createTime.H"

	/* Load meshes and create variables */
	#include "createMeshAndField.H"

	/* Initialize */
	#include "init.H"

	/* compressibleCreatePhi.H */
	rhoUSn = Foam::fvc::interpolate(rhoU) & mesh_gas.Sf();
	update_boundaryField(mesh_gas, rho, U, rhoUSn);

	while(runTime.loop())
	{
		const Foam::dimensionedScalar dt(Foam::dimTime, runTime.deltaTValue());
		Foam::Info << "\nn=" << runTime.timeIndex() << ", t=" << std::stod(runTime.timeName(), nullptr)*s2ms << "ms, dt=" << dt.value()*s2ns << "ns" << Foam::endl;
		runTime.write();

		/* Update gas-phase @(n) with Tw @(n) as Dirichlet B.C. and sources terms @(n) */
		{
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
							Foam::fvm::ddt(rho, U_star) + Foam::fvc::div(rhoUSn, U_next) == -grad_p + 0.5*(Foam::fvm::laplacian(mu, U_star)+Foam::fvc::laplacian(mu, U_next))
						);
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
