"""
This script demonstrates the use of a periodic solution space on a geometry
that is not periodic, by solving the 1D wave equation on the unit interval with
periodic boundary conditions.
"""

from tIGAr.BSplines import *
from tIGAr.timeIntegration import *

####### Parameters #######

# Number of elements:
NEL = 256
# Polynomial degree of basis:
p = 2
# Length of domain:
L = 1.0
# Directory to store ParaView output in:
RESULT_DIR = "results"
# Number of time steps to skip between ParaView files:
OUT_SKIP = 5
# Time $T$ at which to end the computation:
TIME_INTERVAL = 2.0
# Number of time steps:
N_STEPS = NEL

####### Spline setup #######

# Note that the control mesh defining the geometry is NOT periodic.
splineMesh = ExplicitBSplineControlMesh([p,], [uniformKnots(p, 0.0, L, NEL),])

# A field with periodicity is then defined on the same parametric space:
field = BSpline([p,], [uniformKnots(p, 0.0, L, NEL, periodic=True),])

# The field is then combined with the control mesh as a `FieldListSpline`:
splineGenerator = FieldListSpline(splineMesh,[field,])

####### Analysis #######

# Choose the quadrature degree to be used throughout the analysis.
QUAD_DEG = 2*p

# Create an extracted spline:
spline = ExtractedSpline(splineGenerator, QUAD_DEG)

# Displacement solution:
u = Function(spline.V)
u_old = Function(spline.V)
udot_old = Function(spline.V)
uddot_old = Function(spline.V)

# Re-name for output:
u_old.rename("u","u")
udot_old.rename("v","v")

# Create a generalized-alpha time integrator for the unknown field.
RHO_INF = Constant(1.0)
DELTA_T = Constant(TIME_INTERVAL/float(N_STEPS))
timeInt = GeneralizedAlphaIntegrator(RHO_INF, DELTA_T, u,
                                     (u_old, udot_old, uddot_old))

# Get alpha-level quantities to form residual.
u_alpha = timeInt.x_alpha()
udot_alpha = timeInt.xdot_alpha()
uddot_alpha = timeInt.xddot_alpha()

# Test function:
w = TestFunction(spline.V)

# Problem residual:
res = inner(uddot_alpha,w)*spline.dx + inner(spline.grad(u_alpha),
                                             spline.grad(w))*spline.dx
# Exact solution:
def exact(t):
    x = spline.parametricCoordinates()[0] - t
    return sin(2.0*pi*x)

# Project initial condition.
LUMP = True
u_old.assign(spline.project(exact(0.0), rationalize=False, lumpMass=LUMP))
tau = variable(Constant(0.0))
exactdot = diff(exact(tau),tau)
udot_old.assign(spline.project(exactdot, rationalize=False, lumpMass=LUMP))
exactddot = diff(exactdot,tau)
uddot_old.assign(spline.project(exactddot, rationalize=False, lumpMass=LUMP))

# Time stepping loop:
uFile = File(RESULT_DIR+"/u.pvd")
for i in range(0,N_STEPS):
    print("------ Time step "+str(i)+" -------")
    spline.solveNonlinearVariationalProblem(res, derivative(res, u), u)
    if(i%OUT_SKIP == 0):
        uFile << u_old
    timeInt.advance()
