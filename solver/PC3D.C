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
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include <limits>
#include <string>

bool isEqual(double x, double y)
{
  return std::abs(x-y) <= 1e-6 * std::abs(x);
}

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

/* Classification of gas-phase cells */
const Foam::scalar cFluid = 1.0;
const Foam::scalar cIB = 0.0;
const Foam::scalar cSolid = -1.0;

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

bool pnt_inSolid(const Foam::vector &c)
{
	const Foam::vector c0(0.5, 0.5, 0.5);
	const Foam::scalar r = 0.15;

	return Foam::mag(c-c0) < r;
}

bool cell_isIntersected(const Foam::vectorList &v)
{
	bool flag = pnt_inSolid(v[0]);
	for (int i = 1; i < v.size(); i++)
	{
		if (pnt_inSolid(v[i]) != flag)
			return true;
	}
	return false;
}

Foam::scalar cellNum(const Foam::vectorList &v, const Foam::vector &c)
{
	if (pnt_inSolid(c))
		return cSolid;
	else
	{
		if (cell_isIntersected(v))
			return cIB;
		else
			return cFluid;
	}
}

struct ibExtStencil
{
	Foam::label my_cI;
	Foam::label my_proc;
	Foam::label nAdjC;
	Foam::labelList proc;
	Foam::labelList cI;

	void reset()
	{
		my_cI = 0;
		my_proc = 0;
		nAdjC = 0;
		proc.clear();
		cI.clear();
	}
};

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

	Foam::List<ibExtStencil> ibInterpInfo(mesh_gas.nCells());

	/* Classify mesh cells */
	{
		for (int i = 0; i < mesh_gas.nCells(); i++)
		{
			const auto &c = mesh_gas.C()[i];

			const auto &cP = mesh_gas.cellPoints()[i];
			Foam::vectorList p(cP.size());
			for (int j = 0; j < p.size(); j++)
				p[j] = mesh_gas.points()[cP[j]];

			cMarker[i] = cellNum(p, c);
		}

		for (int i = 0; i < mesh_gas.nCells(); i++)
		{
			if (isEqual(cMarker[i], cSolid))
			{
				const auto &cC = mesh_gas.cellCells()[i];
				for (int j = 0; j < cC.size(); j++)
				{
					const auto adjCI = cC[j];
					if (isEqual(cMarker[adjCI], cFluid))
						cMarker[adjCI] = cIB;
				}
			}
		}

		for (int i = 0; i < mesh_gas.nCells(); i++)
		{
			if (cMarker[i] > 0)
				cIbMask[i] = 1.0;
			else
				cIbMask[i] = 0.0;
		}

		for (int i = 0; i < mesh_gas.nCells(); i++)
		{
			auto &e = ibInterpInfo[i];
			e.my_cI = i;
			e.my_proc = Foam::Pstream::myProcNo();

			// Only check cells represents the immersed boundary.
			if (!isEqual(cMarker[i], cIB))
				continue;

			const auto &adjCell = mesh_gas.cellCells()[i];
			const auto &encFace = mesh_gas.cells()[i];
			const auto &incNode = mesh_gas.cellPoints()[i];

			Foam::labelHashSet extCell;

			for (int j = 0; j < encFace.size(); j++)
			{
				const auto fI = encFace[j];
				if (mesh_gas.isInternalFace(fI))
				{

				}
			}
		}
	}

	while(runTime.loop())
	{
		const Foam::dimensionedScalar dt(Foam::dimTime, runTime.deltaTValue());
		Foam::Info << "\nn=" << runTime.timeIndex() << ", t=" << std::stod(runTime.timeName(), nullptr)*s2ms << "ms, dt=" << dt.value()*s2ns << "ns" << Foam::endl;
		runTime.write();

		/* Interpolation on IB cells */
		{
			for (int i = 0; i < mesh_gas.nCells(); i++)
			{
				if (isEqual(cMarker[i], cIB))
				{
					U[i].x() = 0.0;
					U[i].y() = 0.0;
					U[i].z() = 0.0;
				}
			}
		}

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
							Foam::fvm::ddt(rho, U_star) + Foam::fvc::div(rhoUSn, U_next) == -cIbMask * grad_p + 0.5*(Foam::fvm::laplacian(mu, U_star)+Foam::fvc::laplacian(mu, U_next))
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
					for (int i=0; i < mesh_gas.nCells(); i++)
					{
						if (cMarker[i] < cFluid)
							dpEqn.source()[i] = 0.0;
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
