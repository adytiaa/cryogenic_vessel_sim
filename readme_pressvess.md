# Pressure Vessel POD + Neural Network Surrogate (Small Mesh Demo)

## Overview

This project demonstrates an **open-source** workflow for simulating and building a **physics-informed AI surrogate** for a cylindrical pressure vessel under internal pressure.  
We replace commercial FEA tools like ANSYS with **FEniCS** (for physics-based simulation) and combine it with **Proper Orthogonal Decomposition (POD)** and a **Neural Network** for rapid prediction of deformation fields.

The pipeline:
1. Generate a **small 3D hollow cylinder mesh** with **Gmsh**.
2. Run **FEniCS** to solve linear elasticity with varying parameters (Young’s modulus `E` and pressure `p`).
3. Build a **snapshot matrix** from displacement results.
4. Apply **POD** to extract a low-dimensional basis.
5. Train a **PyTorch NN** to map `(E, p)` → POD coefficients.
6. Reconstruct the displacement field from NN predictions.
7. **Compare** NN results vs. FEniCS results with:
   - Error histograms
   - 3D scatter plots of spatial error
   - ParaView `.xdmf` output for full visualization.

This example uses a **coarse mesh** to keep runtimes low (~seconds per simulation), but can be adapted to high-resolution meshes for production.

---

## Files

- `pressure_vessel_pod_nn.ipynb` – Complete runnable Jupyter notebook with explanations.
- `pressure_vessel_pod_nn.py` – Script version (same logic, easier CLI execution).
- `requirements.txt` – Python dependencies.
- *(Generated after running the notebook)*:
  - `vessel.msh`, `vessel_mesh.xdmf`, `vessel_facets.xdmf` – Mesh files.
  - `ground_truth.xdmf` – FEniCS displacement field for chosen parameters.
  - `nn_prediction.xdmf` – Neural network predicted displacement field.
  - `error_field.xdmf` – Absolute displacement error magnitude field.

---

## Dependencies

Install everything with:
```bash
pip install -r requirements.txt
```

Requirements:
- `fenics==2019.1.0` – Physics-based finite element solver.
- `gmsh` – Mesh generation.
- `meshio` – Mesh format conversion.
- `torch` – Neural network training.
- `numpy`, `scikit-learn`, `matplotlib` – Data handling and visualization.

---

## How to Run

### 1. Jupyter Notebook
```bash
jupyter notebook pressure_vessel_pod_nn.ipynb
```
Run all cells to:
- Generate mesh
- Solve PDE for multiple `(E, p)` combinations
- Perform POD
- Train NN
- Output ParaView-ready results

### 2. Python Script
```bash
python pressure_vessel_pod_nn.py
```
This will execute the same pipeline without interactive plots.

---

## Visualization in ParaView

Three `.xdmf` files are saved for ParaView:
- **Ground Truth**: `ground_truth.xdmf`
- **NN Prediction**: `nn_prediction.xdmf`
- **Error Field**: `error_field.xdmf`

To view:
1. Open `.xdmf` file in ParaView and click *Apply*.
2. Use "Surface" or "Warp By Vector" to view displacements.
3. Color by displacement magnitude or error magnitude.
4. Compare ground truth vs NN visually.

---

## Notes

- This is a **demo mesh** with coarse resolution for fast runs.  
  Replace `lc` in the mesh generation step for finer accuracy.
- Extensible to nonlinear materials, thermal loading, or dynamic simulations.
- The POD+NN framework can be swapped for other reduced-order models (e.g., GP regression).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---



        ┌───────────────┐
        │  Geometry &   │
        │   Meshing     │
        │   (Gmsh)      │
        └───────┬───────┘
                │ .xdmf mesh
                ▼
        ┌───────────────┐
        │  Simulation    │
        │  (FEniCS FEM) │
        └───────┬───────┘
                │ displacement snapshots
                ▼
        ┌───────────────┐
        │   POD Basis   │
        │ (SVD on data) │
        └───────┬───────┘
                │ reduced coefficients
                ▼
        ┌───────────────┐
        │ Surrogate ML  │
        │   (POD+PINN)  │
        └───────┬───────┘
                │ predicted coeffs
                ▼
        ┌───────────────┐
        │ Reconstruction │
        │ (POD basis * c)│
        └───────┬────────┘
                │ displacement field
                ▼
        ┌───────────────┐
        │ Visualization │
        │ Matplotlib /  │
        │   ParaView    │
        └───────────────┘


The goal: simulating a pressure vessel using open-source tools like Gmsh, FEniCS, and building a PINN with PyTorch. I'll emphasize the PINN's difference: training directly on elasticity PDEs rather than pre-computed data. Necessary imports are already prepared. Part 1, the Gmsh mesh generation, will use a function for robust geometry definition with physical groups. This function will save the mesh in a FEniCS-compatible format, with clear calls to the function included.

Defining the FEniCS Solver: developing the FEniCS ground truth simulation. I've designed the solve_with_fenics function, ensuring it reads the Gmsh-generated mesh and boundary data, sets up the standard linear elasticity problem, and defines the crucial boundary conditions for pressure and fixed support. The function will also calculate and return the von Mises stress for thorough analysis. My next step involves testing this solver with a specific pressure and visualizing the displacement and stress fields.

Implementing Loss Functions (PINN's loss function): PDE loss using automatic differentiation to enforce the Navier-Cauchy equations, and boundary condition losses for the fixed support and pressure. I'm leveraging PyTorch's autograd.grad for derivative calculations. I'll thoroughly test each loss term individually to ensure correctness.

Developing the FEniCS Solver: Immersed in developing the FEniCS solver, a crucial step. I've successfully designed the solve_with_fenics function, ensuring it accurately reads the Gmsh-generated mesh and boundary data. This solver, complete with boundary conditions for pressure and fixed support, will serve as my ground truth. The implementation includes calculating and returning the von Mises stress for a thorough analysis.

Constructing the PINN Loss Function: Focusing on defining the crucial pinn_loss_function. I'm developing the PDE residual term using torch.autograd.grad to enforce the Navier-Cauchy equations. Simultaneously, I'm working on the boundary condition losses, ensuring accurate implementation for both the fixed support and pressure-loaded boundaries. The total loss will be a weighted sum of these components.
