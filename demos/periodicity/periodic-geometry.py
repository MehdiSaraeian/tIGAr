from tIGAr.BSplines import *
import math

"""
Set up a circular cylinder approximated by a B-spline that is smoothly
periodic in the angular direction, then solve the surface Poisson equation
on it.  This demonstrates both periodicity and usage of the legacy ASCII
format for NURBS patches.
"""

# Radius and length of cylinder:
r = 1.0
L = 4.0

# Degrees in u and v directions:
p = 2 # (Note: Control point formulas assume p=2 when approximating cylinder)
q = 2
# Numbers of elements in u and v directions. Only the number of unique elements
# are given in the periodic u direction, with no duplicate knot spans.
n_el_u = 16
n_el_v = 16

# Create a univariate B-spline for the axial (v/z) direction with an open knot
# vector and control points spaced to produce equal-sized elements in physical
# space.
knots_v = uniformKnots(q,0.0,L,n_el_v)
spline_mesh = ExplicitBSplineControlMesh([q,],[knots_v,])
spline_generator = EqualOrderSpline(1,spline_mesh)
n_cp_v = spline_mesh.getScalarSpline().getNcp()

# Fill in u-direction knot vector with even spacing for unique knots. The first
# and last knot are identified with each other, and periodic continuity is
# determined by their multiplicity. For maximally-continuous periodicity, both
# have a multiplicity of 1 (and for no periodicity, they have multiplicity
# p+1, i.e., discontinuity, or an open knot vector).
knots_u = uniformKnots(p, 0.0, 1.0, n_el_u, periodic=True)

# Calculating control points for tensor-product spline to approximate cylinder:

# Angle between control points, assuming u direction is angular:
dtheta = 2.0*math.pi/float(n_el_u)
# Radius for control points to approximate cylinder with unit weights and p=2:
R = r*math.sqrt(1 + math.tan(0.5*dtheta)**2)
cp_list = []
for j in range(0,n_cp_v):
    # Get z-coordinates of control points from `spline_mesh` to ensure equal
    # sized elements.
    z = spline_mesh.getHomogeneousCoordinate(j,0)
    for i in range(0,n_el_u):
        theta = i*dtheta
        x = R*math.cos(theta)
        y = R*math.sin(theta)
        cp_list += [(x,y,z),]

# Write out a file in the legacy ASCII format:
fname_prefix = "patch."
fname_suffix = ".dat"
# Dimension of physical space:
f_str = "3\n"
# Degrees in each parametric direction:
f_str += str(p) + " " + str(q) + "\n"
# Numbers of control points in each parametric direction:
f_str += str(n_el_u) + " " + str(n_cp_v) + "\n"
# The u and v knot vectors, which determine continuity, with continuity at
# ends understood to mean periodicity:
for k in knots_u:
    f_str += str(k) + " "
f_str += "\n"
for k in knots_v:
    f_str += str(k) + " "    
f_str += "\n"
# The grid of control points built up earlier:
for c in cp_list:
    for i in range(0,3):
        f_str += str(c[i]) + " "
    f_str += "1.0\n"
fname = fname_prefix + "1" + fname_suffix
f = open(fname,"w")
f.write(f_str)
f.close()
    
# Read the file back into a tIGAr control mesh:
control_mesh = LegacyMultipatchControlMesh(fname_prefix, 1, fname_suffix)

# Create a scalar spline space for solving PDEs:
spline_generator = EqualOrderSpline(1, control_mesh)

# Add boundary conditions to the spline generator, pinning the solution to
# zero at the ends of the cylinder.
field = 0 # The single scalar solution field.
scalar_spline = spline_generator.getScalarSpline(field)
parametric_direction = 1 # The v direction.
for side in [0,1]: # The two ends of the cylinder in the v direction.
    # The legacy multi-patch spline has only a single B-spline in this case,
    # whose DoF ordering coincides with the overall spline DoF ordering.
    side_dofs = scalar_spline.splines[0].getSideDofs(parametric_direction, side)
    spline_generator.addZeroDofs(field, side_dofs)

# Generate the extracted representation of the spline. The displacement space
# associated with this spline should have the same periodicity as the geometry,
# because it is using the same function space.
spline = ExtractedSpline(spline_generator, p+q)

# Set up and solve the surface Poisson equation, directly using homogeneous
# form of functions due to use of uniform unit weights.
u = TrialFunction(spline.V)
v = TestFunction(spline.V)
a = inner(spline.grad(u), spline.grad(v))*spline.dx
L = inner(Constant(1.0), v)*spline.dx
u_h = Function(spline.V)
spline.solveLinearVariationalProblem(a==L, u_h)

# Output to ParaView for visualization:

# Output solution function:
u_h.rename("u", "u")
results_dir = "results/"
File(results_dir+"u.pvd") << u_h
# Output the control mesh geometry functions for a qualitative check:
spline.cpFuncs[0].rename("F0","F0")
spline.cpFuncs[1].rename("F1","F1")
spline.cpFuncs[2].rename("F2","F2")
# (No need for weight function here, since it is uniformly 1)
File(results_dir+"F-x.pvd") << spline.cpFuncs[0]
File(results_dir+"F-y.pvd") << spline.cpFuncs[1]
File(results_dir+"F-z.pvd") << spline.cpFuncs[2]

# Visualization in ParaView: Combine data from the four files with an
# Append Attributes filter, then use the calculator filter
#
#  (F0-coordsX)*iHat + (F1-coordsY)*jHat + (F2-coordsZ)*kHat
#
# to Warp by Vector, where the coefficients do not need to be divided through
# by a weight here, because it is a B-spline geometry. This may require
# manualy switching from "2D" to "3D" mode, since ParaView may load the 2D
# parametric-space mesh in 2D mode by default.  Then use th `u` field to color
# this warped geometry, to visualize the PDE solution.
