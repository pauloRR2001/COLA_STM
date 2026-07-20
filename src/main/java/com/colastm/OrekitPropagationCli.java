package com.colastm;

import java.io.File;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.linear.RealMatrix;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.attitudes.LofOffset;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.forces.drag.DragForce;
import org.orekit.forces.drag.IsotropicDrag;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.forces.maneuvers.ConstantThrustManeuver;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.LOFType;
import org.orekit.models.earth.atmosphere.SimpleExponentialAtmosphere;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.MatricesHarvester;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

/** Command-line Orekit propagator used by the Python challenge scripts. */
public final class OrekitPropagationCli {

    private OrekitPropagationCli() { }

    public static void main(final String[] args) {
        Locale.setDefault(Locale.US);
        if (args.length != 18 && args.length != 19) {
            throw new IllegalArgumentException("Expected 18 or 19 arguments, received " + args.length);
        }
        configureOrekitData();

        int k = 0;
        final Vector3D position = new Vector3D(Double.parseDouble(args[k++]), Double.parseDouble(args[k++]), Double.parseDouble(args[k++]));
        final Vector3D velocity = new Vector3D(Double.parseDouble(args[k++]), Double.parseDouble(args[k++]), Double.parseDouble(args[k++]));
        final double mass = Double.parseDouble(args[k++]);
        final double duration = Double.parseDouble(args[k++]);
        final double outputStep = Double.parseDouble(args[k++]);
        final double area = Double.parseDouble(args[k++]);
        final double cd = Double.parseDouble(args[k++]);
        final double rho0 = Double.parseDouble(args[k++]);
        final double h0 = Double.parseDouble(args[k++]);
        final double scaleHeight = Double.parseDouble(args[k++]);
        final double thrust = Double.parseDouble(args[k++]);
        final double isp = Double.parseDouble(args[k++]);
        final String thrustDirection = args[k++].toUpperCase(Locale.ROOT);
        final double epochOffset = Double.parseDouble(args[k++]);
        final boolean outputStm = args.length > k && Boolean.parseBoolean(args[k]);

        final Frame inertial = FramesFactory.getGCRF();
        final Frame earthFixed = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
        final AbsoluteDate epoch0 = new AbsoluteDate(2026, 7, 19, 0, 0, 0.0, TimeScalesFactory.getUTC());
        final AbsoluteDate epoch = epoch0.shiftedBy(epochOffset);
        final double mu = Constants.WGS84_EARTH_MU;
        final Orbit initialOrbit = new CartesianOrbit(new PVCoordinates(position, velocity), inertial, epoch, mu);
        final double[][] tolerances = NumericalPropagator.tolerances(1.0, initialOrbit, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator = new DormandPrince853Integrator(0.01, 900.0, tolerances[0], tolerances[1]);
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(initialOrbit, mass));
        final MatricesHarvester harvester = outputStm ?
                propagator.setupMatricesComputation("stm", null, null) : null;

        final NormalizedSphericalHarmonicsProvider gravity = GravityFieldFactory.getNormalizedProvider(8, 8);
        propagator.addForceModel(new HolmesFeatherstoneAttractionModel(earthFixed, gravity));

        final OneAxisEllipsoid earth = new OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, earthFixed);
        final SimpleExponentialAtmosphere atmosphere = new SimpleExponentialAtmosphere(earth, rho0, h0, scaleHeight);
        propagator.addForceModel(new DragForce(atmosphere, new IsotropicDrag(area, cd)));
        // For these LEO cases, Earth harmonics and atmospheric drag dominate.
        // Third-body gravity and SRP are intentionally omitted because they do
        // not materially change the challenge decisions and add substantial
        // cost to long campaign sweeps.

        if (thrust > 0.0 && !"NONE".equals(thrustDirection)) {
            final LofOffset attitude;
            final Vector3D direction;
            switch (thrustDirection) {
                case "PROGRADE" -> { attitude = new LofOffset(inertial, LOFType.TNW); direction = Vector3D.PLUS_I; }
                case "RETROGRADE" -> { attitude = new LofOffset(inertial, LOFType.TNW); direction = Vector3D.MINUS_I; }
                case "RADIAL_OUT" -> { attitude = new LofOffset(inertial, LOFType.QSW); direction = Vector3D.PLUS_I; }
                case "RADIAL_IN" -> { attitude = new LofOffset(inertial, LOFType.QSW); direction = Vector3D.MINUS_I; }
                case "NORMAL_PLUS" -> { attitude = new LofOffset(inertial, LOFType.QSW); direction = Vector3D.PLUS_K; }
                case "NORMAL_MINUS" -> { attitude = new LofOffset(inertial, LOFType.QSW); direction = Vector3D.MINUS_K; }
                default -> throw new IllegalArgumentException("Unsupported thrust direction: " + thrustDirection);
            }
            propagator.addForceModel(new ConstantThrustManeuver(epoch, duration, thrust, isp, attitude, direction));
        }

        final StringBuilder header = new StringBuilder("time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,mass_kg");
        if (outputStm) {
            for (int row = 0; row < 6; ++row) {
                for (int column = 0; column < 6; ++column) {
                    header.append(String.format(Locale.US, ",phi_%d_%d", row, column));
                }
            }
        }
        System.out.println(header);
        final int samples = Math.max(1, (int) Math.ceil(Math.abs(duration) / outputStep));
        for (int j = 0; j <= samples; ++j) {
            final double dt = Math.copySign(Math.min(Math.abs(duration), j * outputStep), duration);
            final SpacecraftState state = propagator.propagate(epoch.shiftedBy(dt));
            final PVCoordinates pv = state.getPVCoordinates(inertial);
            final StringBuilder row = new StringBuilder(String.format(Locale.US,
                    "%.9f,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e",
                    dt, pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ(),
                    pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ(), state.getMass()));
            if (outputStm) {
                final RealMatrix phi = harvester.getStateTransitionMatrix(state);
                for (int matrixRow = 0; matrixRow < 6; ++matrixRow) {
                    for (int matrixColumn = 0; matrixColumn < 6; ++matrixColumn) {
                        row.append(String.format(Locale.US, ",%.12e", phi.getEntry(matrixRow, matrixColumn)));
                    }
                }
            }
            System.out.println(row);
            if (Math.abs(dt - duration) < 1.0e-9) { break; }
        }
    }

    private static void configureOrekitData() {
        final File orekitData = new File(System.getProperty("orekit.data.path", "orekit-data"));
        if (!orekitData.isDirectory()) {
            throw new IllegalStateException("Orekit data directory not found: " + orekitData.getAbsolutePath());
        }
        final DataProvidersManager manager = DataContext.getDefault().getDataProvidersManager();
        manager.addProvider(new DirectoryCrawler(orekitData));
    }
}
