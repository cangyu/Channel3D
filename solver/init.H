/* gas-phase */
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