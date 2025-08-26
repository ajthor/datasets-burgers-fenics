"""
1D Burgers equation with Dirichlet boundary conditions.

Solves: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
where u(x,t) is the velocity field and ν is the kinematic viscosity.

Physical system: Nonlinear advection-diffusion in a 1D domain with zero boundary conditions.
This equation models shock formation and viscous dissipation.
"""

import numpy as np
from torch.utils.data import IterableDataset

import ufl
import dolfinx
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)


def sample_gp_prior(kernel, X, n_samples=1):
    """
    Sample from Gaussian Process prior for random smooth fields.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    K = kernel(X, X)

    prior = np.random.multivariate_normal(
        mean=np.zeros(X.shape[0]),
        cov=K,
        size=n_samples,
    )

    return prior


def sample_gp_posterior(kernel, X, y, xt, n_samples=1):
    """
    Sample from Gaussian Process posterior for smooth random fields.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if xt.ndim == 1:
        xt = xt.reshape(-1, 1)

    K = kernel(X, X)
    Kt = kernel(X, xt)
    Ktt = kernel(xt, xt)

    K_inv = np.linalg.inv(K)

    mu = Kt.T @ K_inv @ y
    cov = Ktt - Kt.T @ K_inv @ Kt

    mu = mu.squeeze()

    posterior = np.random.multivariate_normal(
        mean=mu,
        cov=cov,
        size=n_samples,
    )

    return posterior


class BurgersDataset(IterableDataset):
    """
    Dataset for 1D Burgers equation simulations with periodic boundary conditions.
    Solves: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    """
    def __init__(
        self,
        # Domain parameters
        Lx=2*np.pi,               # Domain length
        Nx=128,                   # Grid points in x
        # PDE parameters
        viscosity=0.01,           # Kinematic viscosity ν
        # Time integration parameters
        stop_sim_time=2.0,        # Final simulation time
        timestep=0.01,            # Time step size
        save_interval=10,         # Save every N time steps
        dtype=np.float64,
    ):
        """
        Dataset for 1D Burgers equation simulations with periodic boundary conditions.
        Solves: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
        
        Args:
            Lx: Domain length in x-direction
            Nx: Number of grid points in x-direction
            viscosity: Kinematic viscosity coefficient ν
            stop_sim_time: Final simulation time
            timestep: Time step size
            save_interval: Save solution every N time steps
            dtype: Data type for computations
        """
        super().__init__()
        
        # Store domain and grid parameters
        self.Lx = Lx
        self.Nx = Nx
        
        # Store PDE parameters
        self.viscosity = viscosity
        
        # Store time integration parameters
        self.stop_sim_time = stop_sim_time
        self.timestep = timestep
        self.save_interval = save_interval
        self.dtype = dtype
        
        # Create DOLFINx mesh and function space (1D interval)
        self.domain = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, nx=Nx-1)
        self.V = dolfinx.fem.functionspace(self.domain, ("CG", 1))
        
        # Setup Dirichlet boundary conditions (u=0 at boundaries)
        def boundary(x):
            return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        
        self.bc = dolfinx.fem.dirichletbc(
            ScalarType(0), dolfinx.fem.locate_dofs_geometrical(self.V, boundary), self.V
        )
        
        # Get DOF coordinates (scale to domain length)
        self.coordinates = self.domain.geometry.x[:, 0] * Lx
        
        # Define functions for the problem
        self.u_n = dolfinx.fem.Function(self.V)  # Solution at previous time step
        self.u = dolfinx.fem.Function(self.V)    # Solution at current time step

    def __iter__(self):
        """
        Generate infinite samples from the dataset.
        """
        while True:
            # Generate random initial condition using GP
            sigma = 0.2  # GP kernel width
            gamma = 1 / (2 * sigma**2)
            
            # Sample from GP with zero BCs at both ends (unit interval)
            nx = self.domain.geometry.x.shape[0]
            u_init = sample_gp_posterior(
                kernel=partial(rbf_kernel, gamma=gamma),
                X=np.array([0, 1]),
                y=np.array([0, 0]),
                xt=np.linspace(0, 1, nx),
                n_samples=1,
            )[0]
            
            # Interpolate to mesh points
            margin = 1e-3
            interp_function = interp1d(
                np.linspace(-margin, 1 + margin, nx),
                u_init,
                kind="cubic",
            )
            
            # Set initial condition on the function
            self.u.interpolate(lambda x: interp_function(x[0]))
            u_init_array = self.u.x.array.copy()
            
            # Random viscosity parameter
            viscosity = np.random.uniform(0.01, 0.1)
            
            # Solve the PDE and yield result
            yield self.solve(u_init_array, viscosity)

    def solve(self, initial_condition, viscosity):
        """
        Solve the Burgers equation for a given initial condition and viscosity.

        Args:
            initial_condition: Initial condition u(x,0) as a numpy array.
            viscosity: Viscosity parameter for this solve.

        Returns:
            A dictionary containing all data useful for learning the PDE.
        """
        
        # Set initial condition
        self.u_n.x.array[:] = initial_condition
        self.u.x.array[:] = initial_condition
        
        # Define trial and test functions
        du = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        # Weak form of the time-discretized Burgers equation
        dt = self.timestep
        F = (
            ((self.u - self.u_n) / dt) * v * ufl.dx
            + viscosity * ufl.dot(ufl.grad(self.u), ufl.grad(v)) * ufl.dx
            + self.u * ufl.grad(self.u)[0] * v * ufl.dx
        )
        
        # Jacobian for Newton's method
        J = ufl.derivative(F, self.u, du)
        
        # Time-stepping loop
        n_steps = int(self.stop_sim_time / dt)
        u_trajectory = []
        time_coords = []
        
        # Save initial condition
        u_trajectory.append(self.u.x.array.copy())
        time_coords.append(0.0)
        
        for step in range(n_steps):
            # Nonlinear solve at each time step
            problem = NonlinearProblem(F, self.u, bcs=[self.bc], J=J)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.atol = 1e-8
            solver.rtol = 1e-8
            solver.max_it = 50

            n, converged = solver.solve(self.u)
            
            if not converged:
                logger.warning(f"Newton solver did not converge at step {step}")
            
            # Update previous solution for next time step
            self.u_n.x.array[:] = self.u.x.array
            
            # Save solution at specified intervals
            if step % self.save_interval == 0:
                u_trajectory.append(self.u.x.array.copy())
                time_coords.append((step + 1) * dt)
        
        # Convert to numpy arrays
        u_trajectory = np.array(u_trajectory)
        time_coordinates = np.array(time_coords)

        return {
            # Coordinates
            "spatial_coordinates": self.coordinates,  # Shape: (Nx,)
            "time_coordinates": time_coordinates,     # Shape: (nt,)
            
            # Solution fields
            "u_initial": initial_condition,           # Initial condition: (Nx,)
            "u_trajectory": u_trajectory,             # Full evolution: (nt, Nx)
            
            # PDE parameters
            "viscosity": viscosity,
            "domain_length": self.Lx,
            "timestep": self.timestep,
            "save_interval": self.save_interval,
        }