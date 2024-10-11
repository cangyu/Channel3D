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
    Foam::IOobject("rho", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDensity, 1.0),
    "zeroGradient"
);

Foam::volScalarField rho_star // Provisional density, Unit: kg/m^3
(
    Foam::IOobject("rho_star", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    rho
);

Foam::volVectorField U // Velocity@(n), Unit: m/s
(
    Foam::IOobject("U", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volScalarField p // Pressure@(n), Unit: Pa
(
    Foam::IOobject("p", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volScalarField dp // Pressure incremental correction, Unit: Pa
(
    Foam::IOobject("dp", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::NO_WRITE),
    mesh_gas
);

Foam::volScalarField T // Temperature@(n), Unit: K
(
    Foam::IOobject("T", runTime.timeName(), mesh_gas, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas
);

Foam::volScalarField mu // Viscosity, Unit: kg/m/s
(
    Foam::IOobject("mu", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDynamicViscosity, 0.0),
    "zeroGradient"
);

Foam::volScalarField Cp // Constant-pressure specific heat, Unit: J/kg/K
(
    Foam::IOobject("Cp", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimSpecificHeatCapacity, 0.0),
    "zeroGradient"
);

Foam::volScalarField lambda // Conductivity, Unit: W/m/K
(
    Foam::IOobject("lambda", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimPower/Foam::dimLength/Foam::dimTemperature, 0.0),
    "zeroGradient"
);

/* Gradients */
Foam::volVectorField grad_p
(
    Foam::IOobject("grad_p", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedVector(Foam::dimPressure/Foam::dimLength, Foam::vector(0, 0, 0))
);

/* Surface variables */
Foam::surfaceScalarField rhoUSn // Mass flux, Unit: kg/s
(
    Foam::IOobject("rhoUSn", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDensity*Foam::dimVelocity*Foam::dimArea, 0.0)
);

Foam::surfaceScalarField USn // Volume flux, Unit: m^3/s
(
    Foam::IOobject("USn", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimVelocity*Foam::dimArea, 0.0)
);

/* Thermophysical properties */
Foam::autoPtr<Foam::fluidThermo> pThermo
(
    Foam::fluidThermo::New(mesh_gas)
);
Foam::fluidThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

/* Turbulence model */
Foam::autoPtr<Foam::compressible::momentumTransportModel> turbulence
(
    Foam::compressible::momentumTransportModel::New(rho, U, rhoUSn, thermo)
);
turbulence->validate();
