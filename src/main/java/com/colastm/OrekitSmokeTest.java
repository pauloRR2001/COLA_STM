package com.colastm;

import java.io.File;

import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.orekit.data.DataContext;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;

public class OrekitSmokeTest {
    public static void main(String[] args) {
        configureOrekitData();

        final double earthRadius = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
        final double mu = Constants.WGS84_EARTH_MU;
        final Frame inertial = FramesFactory.getEME2000();
        final AbsoluteDate epoch = new AbsoluteDate(2026, 7, 19, 0, 0, 0.0, TimeScalesFactory.getUTC());

        final double altitudeM = 330_000.0;
        final double a = earthRadius + altitudeM;
        final double e = 0.0;
        final double i = Math.toRadians(51.6);
        final double raan = 0.0;
        final double argPerigee = 0.0;
        final double trueAnomaly = 0.0;

        final Orbit initialOrbit = new KeplerianOrbit(
            a, e, i, argPerigee, raan, trueAnomaly,
            PositionAngleType.TRUE, inertial, epoch, mu
        );

        final double minStep = 0.001;
        final double maxStep = 300.0;
        final double positionTolerance = 1.0;
        final double[][] tolerances = NumericalPropagator.tolerances(positionTolerance, initialOrbit, OrbitType.CARTESIAN);
        final DormandPrince853Integrator integrator = new DormandPrince853Integrator(
            minStep, maxStep, tolerances[0], tolerances[1]
        );

        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(OrbitType.CARTESIAN);
        propagator.setInitialState(new SpacecraftState(initialOrbit, 420.0));

        // This smoke test intentionally uses only two-body dynamics.
        // Challenge-specific force models will be added next: J2, drag, finite HET thrust, and mass depletion.
        final SpacecraftState finalState = propagator.propagate(epoch.shiftedBy(3600.0));

        System.out.printf("Orekit smoke test OK%n");
        System.out.printf("Initial altitude: %.3f km%n", altitudeM / 1000.0);
        System.out.printf("Altitude after 1 hour, two-body only: %.6f km%n",
            (finalState.getOrbit().getA() - earthRadius) / 1000.0);
        System.out.printf("Mass: %.3f kg%n", finalState.getMass());
    }

    private static void configureOrekitData() {
        final String dataPath = System.getProperty("orekit.data.path", "orekit-data");
        final File orekitData = new File(dataPath);
        if (!orekitData.exists() || !orekitData.isDirectory()) {
            throw new IllegalStateException(
                "Orekit data directory not found: " + orekitData.getAbsolutePath() + System.lineSeparator() +
                "Download orekit-data and run with -Dorekit.data.path=<path> if needed."
            );
        }
        final DataProvidersManager manager = DataContext.getDefault().getDataProvidersManager();
        manager.addProvider(new DirectoryCrawler(orekitData));
    }
}
