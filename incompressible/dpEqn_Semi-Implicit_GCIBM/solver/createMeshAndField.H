/* Mesh of gas-phase */
Foam::fvMesh mesh_gas
(
    Foam::IOobject("gas", runTime.timeName(), runTime, Foam::IOobject::MUST_READ)
);

const auto& bMesh_gas = mesh_gas.boundaryMesh();

Foam::globalMeshData meshInfo_gas(mesh_gas);

/* Volume variables of gas-phase */
Foam::volScalarField rho // Density@(n), Unit: kg/m^3
(
    Foam::IOobject("rho", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volVectorField U // Velocity@(n), Unit: m/s
(
    Foam::IOobject("U", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volVectorField U_star // Provisional velocity@(n), Unit: m/s
(
    Foam::IOobject("U_star", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    U
);

Foam::volVectorField U_next // Velocity@(m), Unit: m/s
(
    Foam::IOobject("U_next", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    U
);

Foam::volScalarField p // Pressure@(n), Unit: Pa
(
    Foam::IOobject("p", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volScalarField p_next // Pressure@(m), Unit: Pa
(
    Foam::IOobject("p_next", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    p
);

Foam::volScalarField dp // Pressure incremental correction, Unit: Pa
(
    Foam::IOobject("dp", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
    mesh_gas
);

Foam::volVectorField rhoU // Momentum@(n), Unit: kg/m^2/s
(
    Foam::IOobject("rhoU", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    rho*U
);

Foam::volVectorField rhoU_star // Provisional momentum, Unit: kg/m^2/s
(
    Foam::IOobject("rhoU_star", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    rhoU
);

Foam::volVectorField rhoU_next // Momentum@(m), Unit: kg/m^2/s
(
    Foam::IOobject("rhoU_next", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    rhoU
);

Foam::volScalarField mu // Viscosity, Unit: kg/m/s
(
    Foam::IOobject("mu", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDynamicViscosity, Foam::Zero),
    "zeroGradient"
);

Foam::volScalarField cMarker
(
    Foam::IOobject("cMarker", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimless, 0.0)
);

Foam::volScalarField cIbMask
(
    Foam::IOobject("cIbMask", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimless, 1.0)
);

/* Gradients */
Foam::volScalarField div_rhoU
(
    Foam::IOobject("div_rhoU", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDensity*Foam::dimVelocity/Foam::dimLength, Foam::Zero)
);

Foam::volVectorField grad_p
(
    Foam::IOobject("grad_p", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedVector(Foam::dimPressure/Foam::dimLength, Foam::Zero)
);

/* Surface variables */
Foam::surfaceScalarField rhoUSn // Mass flow rate, Unit: kg/s
(
    Foam::IOobject("rhoUSn", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDensity*Foam::dimVelocity*Foam::dimArea, Foam::Zero)
);

Foam::surfaceScalarField rhoUSn_star // Provisional mass flow rate, Unit: kg/s
(
    Foam::IOobject("rhoUSn_star", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    rhoUSn
);
