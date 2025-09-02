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
    
    Args:
        kernel: Kernel function (e.g., from sklearn.metrics.pairwise)
        X: Input points, shape (N,) or (N, d)
        n_samples: Number of samples to generate
        
    Returns:
        Array of shape (n_samples, N) containing GP samples
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
    
    This generates smooth random functions that satisfy given boundary conditions
    by conditioning a GP on observed values at specific points.
    
    Args:
        kernel: Kernel function (e.g., from sklearn.metrics.pairwise)
        X: Observed input points, shape (M,) or (M, d)
        y: Observed output values, shape (M,) or (M, 1)
        xt: Target input points, shape (N,) or (N, d)
        n_samples: Number of samples to generate
        
    Returns:
        Array of shape (n_samples, N) containing GP posterior samples
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
    Dataset for 1D viscous Burgers equation simulations with Dirichlet boundary conditions.
    
    Solves: ∂u/∂t + u∂u/∂x = ν∂²u/∂x² on x ∈ [0, 1], t ∈ [0, T]
    
    Boundary conditions: u(0, t) = u(1, t) = 0
    Initial condition: u(x, 0) = u₀(x) (sampled from Gaussian process)
    
    Each sample generates:
    - Random smooth initial condition satisfying zero boundary conditions
    - Random viscosity parameter ν ∈ [0.01, 0.1]
    - Full space-time evolution trajectory for operator learning
    
    The equation models nonlinear wave steepening (shock formation) balanced by
    viscous diffusion, making it ideal for studying nonlinear PDE dynamics.
    """
    def __init__(
        self,
        # Domain parameters
        Lx=2*np.pi,               # Physical domain length (for scaling)
        Nx=128,                   # Number of grid points
        # Time integration parameters  
        stop_sim_time=2.0,        # Final simulation time
        timestep=0.01,            # Time step size
        save_interval=10,         # Save solution every N time steps
        # Solver parameters
        dtype=np.float64,         # Data type for computations
    ):
        """
        Initialize the 1D Burgers equation dataset generator.
        
        Args:
            Lx: Physical domain length (used for coordinate scaling)
            Nx: Number of spatial grid points 
            stop_sim_time: Final simulation time T
            timestep: Time integration step size Δt
            save_interval: Save solution every N time steps (for trajectory)
            dtype: Numerical data type for computations
            
        Note:
            The computational domain is always [0,1], scaled to physical length Lx
            for coordinate output. Viscosity ν is randomly sampled for each sample.
        """
        super().__init__()
        
        # Store domain and grid parameters
        self.Lx = Lx
        self.Nx = Nx
        
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
        Generate infinite samples from the Burgers equation dataset.
        
        Each iteration yields a dictionary containing:
        - spatial_coordinates: Physical x coordinates, shape (Nx,)
        - time_coordinates: Time points, shape (nt,)
        - u_initial: Initial condition u₀(x), shape (Nx,)
        - u_trajectory: Full solution u(x,t), shape (nt, Nx)
        - viscosity: Random viscosity parameter ν for this sample
        - domain_length: Physical domain length
        - timestep, save_interval: Solver parameters
        
        Yields:
            dict: Sample data dictionary for operator learning
        """
        while True:
            # Generate random smooth initial condition with zero boundary conditions
            sigma = 0.2  # GP kernel width (controls smoothness)
            gamma = 1 / (2 * sigma**2)
            
            # Sample from GP conditioned on zero values at boundaries
            nx = self.domain.geometry.x.shape[0]
            u_init = sample_gp_posterior(
                kernel=partial(rbf_kernel, gamma=gamma),
                X=np.array([0, 1]),  # Boundary points
                y=np.array([0, 0]),  # Zero boundary conditions
                xt=np.linspace(0, 1, nx),  # Mesh points on unit interval
                n_samples=1,
            )[0]
            
            # Smooth interpolation to mesh points (with small margin for stability)
            margin = 1e-3
            interp_function = interp1d(
                np.linspace(-margin, 1 + margin, nx),
                u_init,
                kind="cubic",
                bounds_error=False,
                fill_value=0.0
            )
            
            # Set initial condition on FEniCS function
            self.u.interpolate(lambda x: interp_function(x[0]))
            u_init_array = self.u.x.array.copy()
            
            # Generate random viscosity parameter for this sample
            # viscosity = np.random.uniform(0.01, 0.1)
            viscosity = 0.1
            
            # Solve the PDE and yield result
            yield self.solve(u_init_array, viscosity)

    def solve(self, initial_condition, viscosity):
        """
        Solve the 1D viscous Burgers equation using FEniCS and Newton's method.
        
        Uses backward Euler time integration with Newton iteration to handle
        the nonlinear advection term u∂u/∂x.

        Args:
            initial_condition: Initial condition u₀(x), shape (Nx,)
            viscosity: Viscosity parameter ν for this solve

        Returns:
            dict: Solution data containing:
                - spatial_coordinates: Physical x coordinates  
                - time_coordinates: Time evolution points
                - u_initial: Initial condition array
                - u_trajectory: Full space-time solution
                - viscosity: Viscosity parameter used
                - domain_length, timestep, save_interval: Problem parameters
        """
        
        # Set initial condition
        self.u_n.x.array[:] = initial_condition
        self.u.x.array[:] = initial_condition
        
        # Define trial and test functions for variational formulation
        du = ufl.TrialFunction(self.V)  # For Jacobian linearization
        v = ufl.TestFunction(self.V)    # Test function
        
        # Weak form of the time-discretized Burgers equation:
        # ∫(u - u_n)/Δt v dx + ∫ν ∇u·∇v dx + ∫u(∂u/∂x)v dx = 0
        dt = self.timestep
        F = (
            ((self.u - self.u_n) / dt) * v * ufl.dx +         # Time derivative
            viscosity * ufl.dot(ufl.grad(self.u), ufl.grad(v)) * ufl.dx +  # Viscous term
            self.u * ufl.grad(self.u)[0] * v * ufl.dx          # Nonlinear advection
        )
        
        # Compute Jacobian for Newton's method: J = ∂F/∂u
        J = ufl.derivative(F, self.u, du)
        
        # Time-stepping with implicit backward Euler scheme
        n_steps = int(self.stop_sim_time / dt)
        u_trajectory = []
        time_coords = []
        
        # Save initial condition
        u_trajectory.append(self.u.x.array.copy())
        time_coords.append(0.0)
        
        for step in range(n_steps):
            # Solve nonlinear system at each time step using Newton's method
            problem = NonlinearProblem(F, self.u, bcs=[self.bc], J=J)
            solver = NewtonSolver(MPI.COMM_WORLD, problem)
            solver.atol = 1e-8     # Absolute tolerance
            solver.rtol = 1e-8     # Relative tolerance  
            solver.max_it = 50     # Maximum Newton iterations

            num_iterations, converged = solver.solve(self.u)
            
            if not converged:
                logger.warning(f"Newton solver did not converge at step {step} "
                              f"(took {num_iterations} iterations)")
            
            # Update previous time step solution
            self.u_n.x.array[:] = self.u.x.array
            
            # Save solution trajectory at specified intervals
            if (step + 1) % self.save_interval == 0:
                u_trajectory.append(self.u.x.array.copy())
                time_coords.append((step + 1) * dt)
        
        # Convert to numpy arrays
        u_trajectory = np.array(u_trajectory)
        time_coordinates = np.array(time_coords)

        return {
            # Coordinate arrays
            "spatial_coordinates": self.coordinates,      # Physical x points, shape (Nx,)
            "time_coordinates": time_coordinates,         # Time points, shape (nt,)
            
            # Solution data
            "u_initial": initial_condition,               # Initial condition u₀(x), shape (Nx,)
            "u_trajectory": u_trajectory,                 # Evolution u(x,t), shape (nt, Nx)
            
            # PDE parameters for this sample
            "viscosity": viscosity,                       # Viscosity coefficient ν
            "domain_length": self.Lx,                     # Physical domain length
            
            # Solver metadata
            "timestep": self.timestep,                    # Time step size used
            "save_interval": self.save_interval,          # Trajectory sampling interval
        }