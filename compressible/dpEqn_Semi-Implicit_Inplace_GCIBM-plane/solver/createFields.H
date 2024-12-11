Foam::Info << "Create field variables" << Foam::nl << Foam::endl;

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

Foam::volScalarField dp // Incremental pressure-correction, Unit: Pa
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

Foam::volScalarField phi // Level-Set function defined as signed-distance, Unit: m
(
    Foam::IOobject("phi", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimLength, 0.0)
);

Foam::volScalarField cIbMarker // Classification of mesh cells
(
    Foam::IOobject("cIbMarker", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimless, 0.0),
    "zeroGradient"
);

Foam::volScalarField cIbMask // Mask of immersed-boundary cells
(
    Foam::IOobject("cIbMask", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimless, 1.0),
    "zeroGradient"
);

/* Gradients */
Foam::volVectorField grad_p
(
    Foam::IOobject("grad_p", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedVector(Foam::dimPressure/Foam::dimLength, Foam::vector(0.0, 0.0, 0.0))
);

/* Surface variables */
Foam::surfaceScalarField rhoUSn // Mass flux, Unit: kg/s
(
    Foam::IOobject("rhoUSn", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDensity*Foam::dimVelocity*Foam::dimArea, 0.0)
);

Foam::surfaceScalarField USn // Volume flux, Unit: m^3/s
(
    Foam::IOobject("USn", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
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
