# ---
# # Pressure Vessel POD + Neural Network Surrogate (Small Mesh Demo)
#
# **Overview:**
# - Build a small hollow cylinder (pressure vessel) mesh with Gmsh.
# - Solve elasticity with internal pressure using FEniCS.
# - Generate multiple solutions (snapshots) for varying material & load parameters.
# - Compute POD basis and project snapshots.
# - Train a small NN surrogate to map parameters → POD coefficients.
# - Reconstruct displacements from NN predictions.
# - Analyze and visualize accuracy (histograms, 3D scatter).
# - Export results to ParaView (ground truth, prediction, error field).
# ---

import gmsh
import meshio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dolfin import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# 1. Generate small mesh with Gmsh
# ---------------------------
gmsh.initialize()
gmsh.model.add("small_vessel")

R = 1.0      # outer radius
L = 3.0      # length
th = 0.1     # wall thickness
lc = 0.3     # coarse mesh size

gmsh.model.occ.addCylinder(0,0,0, R, 0,0,L, tag=1)
gmsh.model.occ.addCylinder(0,0,0, R-th, 0,0,L, tag=2)
gmsh.model.occ.cut([(3,1)], [(3,2)], tag=3)

gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [3], tag=1)
gmsh.model.setPhysicalName(3, 1, "Vessel")

inner_faces = gmsh.model.getBoundary([(3,3)], oriented=False)
gmsh.model.addPhysicalGroup(2, [f[1] for f in inner_faces], tag=2)
gmsh.model.setPhysicalName(2, 2, "InnerWall")

gmsh.model.mesh.generate(3)
gmsh.write("vessel.msh")
gmsh.finalize()

# ---------------------------
# 2. Convert mesh for FEniCS
# ---------------------------
mesh_from_file = meshio.read("vessel.msh")
cells = {"tetra": mesh_from_file.get_cells_type("tetra")}
cell_data = {"name_to_read": [mesh_from_file.cell_data_dict["gmsh:physical"]["tetra"]]}

meshio.write("vessel_mesh.xdmf",
             meshio.Mesh(points=mesh_from_file.points,
                         cells=cells,
                         cell_data=cell_data))

facet_cells = mesh_from_file.get_cells_type("triangle")
facet_data = mesh_from_file.cell_data_dict["gmsh:physical"]["triangle"]
meshio.write("vessel_facets.xdmf",
             meshio.Mesh(points=mesh_from_file.points,
                         cells={"triangle": facet_cells},
                         cell_data={"name_to_read": [facet_data]}))

# ---------------------------
# 3. FEniCS solver
# ---------------------------
def solve_vessel(E, nu, p):
    mesh = Mesh()
    with XDMFFile("vessel_mesh.xdmf") as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("vessel_facets.xdmf") as infile:
        infile.read(mvc, "name_to_read")
    facet_markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    mu = E / (2.0*(1.0+nu))
    lmbda = E*nu / ((1.0+nu)*(1.0-2.0*nu))
    
    def sigma(u):
        return lmbda*tr(sym(grad(u)))*Identity(3) + 2.0*mu*sym(grad(u))
    
    class Clamp(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[2], 0.0, DOLFIN_EPS)
    bc = DirichletBC(V, Constant((0,0,0)), Clamp())
    
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), sym(grad(v))) * dx
    
    n = FacetNormal(mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)
    L = dot(Constant((0,0,p)), v) * ds(2)
    
    u_sol = Function(V)
    solve(a == L, u_sol, bc)
    return u_sol

# ---------------------------
# 4. Generate dataset
# ---------------------------
E_vals = np.linspace(1e9, 5e9, 5)  # Pa
p_vals = np.linspace(1e5, 5e5, 5)  # Pa
nu = 0.3

snapshots = []
params = []

for E in E_vals:
    for p in p_vals:
        u_sol = solve_vessel(E, nu, p)
        snapshots.append(u_sol.vector().get_local())
        params.append([E, p])

snapshots = np.array(snapshots)
params = np.array(params)

# ---------------------------
# 5. POD basis
# ---------------------------
U, S, VT = np.linalg.svd(snapshots - snapshots.mean(0), full_matrices=False)
r = 5
Phi = U[:, :r]
coeffs = snapshots @ Phi

# ---------------------------
# 6. Train NN surrogate
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(params, coeffs, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, r)
)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(500):
    opt.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.6f}")

# ---------------------------
# 7. Test reconstruction
# ---------------------------
pred_coeffs = model(X_test_t).detach().numpy()
recon_snapshots = pred_coeffs @ Phi.T + snapshots.mean(0)

err = np.linalg.norm(recon_snapshots - snapshots[:len(recon_snapshots)], 'fro') / np.linalg.norm(snapshots[:len(recon_snapshots)], 'fro')
print("Relative reconstruction error:", err)

# ---------------------------
# 8. Save ParaView-ready displacement fields
# ---------------------------
def save_displacement_to_xdmf(filename, displacement_vector, mesh):
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    u_field = Function(V)
    u_field.vector().set_local(displacement_vector)
    with XDMFFile(filename) as xdmf:
        xdmf.write(u_field)

mesh = Mesh()
with XDMFFile("vessel_mesh.xdmf") as infile:
    infile.read(mesh)

test_idx = 0
gt_disp = snapshots[test_idx]
nn_disp = recon_snapshots[test_idx]

save_displacement_to_xdmf("ground_truth.xdmf", gt_disp, mesh)
save_displacement_to_xdmf("nn_prediction.xdmf", nn_disp, mesh)
print("✅ Saved ground_truth.xdmf and nn_prediction.xdmf for ParaView comparison.")

# ---------------------------
# 9. Error histogram
# ---------------------------
gt_mag = np.linalg.norm(gt_disp.reshape(-1, 3), axis=1)
nn_mag = np.linalg.norm(nn_disp.reshape(-1, 3), axis=1)
error_mag = np.abs(nn_mag - gt_mag)

plt.figure(figsize=(6,4))
plt.hist(error_mag, bins=30, color='royalblue', alpha=0.7)
plt.xlabel("Absolute displacement magnitude error [m]")
plt.ylabel("Number of DOFs")
plt.title("NN vs. Ground Truth Error Distribution")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Max error: {error_mag.max():.3e} m")
print(f"Mean error: {error_mag.mean():.3e} m")

# ---------------------------
# 10. 3D scatter plot of error field
# ---------------------------
coords = mesh.coordinates()
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(coords[:,0], coords[:,1], coords[:,2],
               c=error_mag, cmap='viridis', s=20)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("Spatial Distribution of Displacement Magnitude Error")
fig.colorbar(p, ax=ax, label="Absolute Error [m]")
plt.show()

# ---------------------------
# 11. Save error field for ParaView
# ---------------------------
def save_error_field(filename, error_array, mesh):
    V = FunctionSpace(mesh, "CG", 1)  # scalar space
    err_func = Function(V)
    err_func.vector().set_local(error_array)
    with XDMFFile(filename) as xdmf:
        xdmf.write(err_func)

save_error_field("error_field.xdmf", error_mag, mesh)
print("✅ Saved error_field.xdmf for ParaView scalar visualization.")

