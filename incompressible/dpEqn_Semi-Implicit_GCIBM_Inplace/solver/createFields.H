Foam::Info << "Create field variables" << Foam::nl << Foam::endl;

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

Foam::volScalarField mu // Viscosity, Unit: kg/m/s
(
    Foam::IOobject("mu", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::NO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimDynamicViscosity, Foam::Zero),
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
    Foam::dimensionedScalar(Foam::dimless, 0.0)
);

Foam::volScalarField cIbMask // Mask of immersed-boundary cells
(
    Foam::IOobject("cIbMask", runTime.timeName(), mesh_gas, Foam::IOobject::NO_READ, Foam::IOobject::AUTO_WRITE),
    mesh_gas,
    Foam::dimensionedScalar(Foam::dimless, 1.0)
);

/* Gradients */
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
