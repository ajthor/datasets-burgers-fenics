"""
TEMPLATE: Dataset class for PDE simulation datasets.

INSTRUCTIONS FOR CLAUDE:
1. Replace this docstring with equation-specific description
2. Add equation in mathematical notation (LaTeX format preferred)
3. Describe the physical system being simulated
4. Import the necessary libraries for your PDE solver below

Example docstrings:
- "2D Heat equation with Dirichlet boundary conditions"
- "1D Wave equation with periodic boundaries"
- "2D Navier-Stokes equations in a lid-driven cavity"
"""

import numpy as np
from torch.utils.data import IterableDataset

# TODO: Import your PDE solver library here
# Examples:
# import dedalus.public as d3  # For spectral methods
# import jax.numpy as jnp; from jax import jit  # For JAX-based solvers
# import torch  # For neural PDE solvers
# from scipy.integrate import solve_ivp  # For scipy ODE solvers

import logging

# TODO: Import additional utilities as needed
# Examples:
# from functools import partial
# from sklearn.metrics.pairwise import rbf_kernel  # For GP initial conditions
# import scipy.sparse as sp  # For sparse matrices

logger = logging.getLogger(__name__)


def sample_gp_prior(kernel, X, n_samples=1):
    """
    Sample from Gaussian Process prior - KEEP if using GP-based initial conditions.
    
    INSTRUCTIONS FOR CLAUDE:
    - This function samples from GP prior (zero mean) for random smooth fields
    - REMOVE this function if your dataset doesn't use GP sampling  
    - KEEP for generating smooth random initial conditions
    - Requires a kernel function (e.g., from sklearn.metrics.pairwise.rbf_kernel)
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
    Sample from Gaussian Process posterior - KEEP if using GP-based initial conditions.

    INSTRUCTIONS FOR CLAUDE:
    - This function is commonly used across datasets for smooth random initial conditions
    - REMOVE this function if your dataset doesn't use GP sampling
    - KEEP if you want smooth, correlated random fields as initial conditions
    - Requires sklearn.metrics.pairwise.rbf_kernel import (or similar kernel function)
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


# TODO: Add additional utility functions as needed for your specific PDE
# Common examples:
# - Vortex generation functions for fluid dynamics
# - Wave packet generators for wave equations
# - Heat source/sink generators for thermal problems


class YourDataset(IterableDataset):
    """
    INSTRUCTIONS FOR CLAUDE:
    1. Rename this class to match your PDE (e.g., HeatEquationDataset, WaveDataset, NavierStokesDataset)
    2. Update this docstring with your equation description and mathematical form
    3. Keep the same __init__ signature pattern but replace parameters with your PDE-specific ones
    """
    def __init__(
        self,
        # TODO: Replace these parameters with your PDE-specific parameters
        # Keep similar structure: domain_size, grid_points, equation_parameters, solver_parameters
        # 
        # Example parameter patterns:
        # Domain parameters:
        Lx=10,                    # Domain length/width (or Lx, Ly for 2D)  
        Nx=1024,                  # Grid points (or Nx, Ny for 2D)
        # 
        # PDE-specific parameters (replace a, b with your equation coefficients):
        # diffusion_coeff=1e-4,   # Diffusion/viscosity coefficient
        # wave_speed=1.0,         # Wave speed for hyperbolic equations
        # source_strength=0.1,    # Source term strength
        # boundary_conditions="periodic",  # Boundary condition type
        # 
        # Solver parameters:
        # dealias=3/2,            # Dealiasing factor (for spectral methods)
        stop_sim_time=10,         # Final simulation time
        timestep=2e-3,           # Time step size  
        # timestepper=...,        # Time integration scheme (solver-specific)
        dtype=np.float64,
    ):
        """
        TODO: Replace this docstring with your PDE description
        
        Template:
        Dataset for [EQUATION_NAME] simulations with [BOUNDARY_CONDITIONS].
        Solves: [MATHEMATICAL_EQUATION_HERE]
        
        Args:
            Lx: Domain length in x-direction
            Nx: Number of grid points in x-direction
            [your_parameter]: Description of your PDE parameter
            stop_sim_time: Final simulation time
            timestep: Time step size
            dtype: Data type for computations
        """
        super().__init__()
        
        # TODO: Store your domain and grid parameters
        self.Lx = Lx
        self.Nx = Nx
        
        # TODO: Store your PDE-specific parameters
        # self.diffusion_coeff = diffusion_coeff
        # self.wave_speed = wave_speed
        # etc.
        
        # Store solver parameters
        self.stop_sim_time = stop_sim_time
        self.timestep = timestep
        self.dtype = dtype
        
        # TODO: Setup your solver components
        # Replace this section with your solver initialization
        # 
        # DEDALUS EXAMPLE (spectral methods):
        # self.xcoord = d3.Coordinate("x")
        # self.dist = d3.Distributor(self.xcoord, dtype=dtype)
        # self.xbasis = d3.RealFourier(self.xcoord, size=Nx, bounds=(0, Lx))
        # self.x = self.dist.local_grid(self.xbasis)
        #
        # FINITE DIFFERENCE EXAMPLE:
        # self.x = np.linspace(0, Lx, Nx)
        # self.dx = Lx / (Nx - 1)
        #
        # JAX EXAMPLE:
        # self.x = jnp.linspace(0, Lx, Nx)
        
        # TODO: Setup your PDE problem/operators
        # Replace with your equation setup
        #
        # DEDALUS EXAMPLE:
        # self.u = self.dist.Field(name="u", bases=self.xbasis)
        # dx = lambda A: d3.Differentiate(A, self.xcoord)
        # self.problem = d3.IVP([self.u], namespace=locals())
        # self.problem.add_equation("dt(u) = diffusion_coeff*dx(dx(u))")  # Heat equation
        #
        # FINITE DIFFERENCE EXAMPLE:
        # self.laplacian_matrix = self._build_laplacian_matrix()
        #
        # CUSTOM/JAX EXAMPLE:
        # self.solve_step = jit(self._time_step)  # JIT compile your solver step
        
        # TODO: Setup spatial coordinates in CONSISTENT FORMAT
        # Examples for different spatial dimensions:
        #
        # 1D case:
        # self.x = np.linspace(0, Lx, Nx) 
        # # Later in solve(): "spatial_coordinates": self.x  # Shape: (Nx,)
        #
        # 2D case:  
        # self.x = np.linspace(0, Lx, Nx)
        # self.y = np.linspace(0, Ly, Ny) 
        # X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        # self.coordinates = np.column_stack((X.ravel(), Y.ravel()))
        # # Later in solve(): "spatial_coordinates": self.coordinates  # Shape: (Nx*Ny, 2)
        #
        # 3D case:
        # self.x, self.y, self.z = [np.linspace(0, L, N) for L, N in zip([Lx, Ly, Lz], [Nx, Ny, Nz])]
        # X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij') 
        # self.coordinates = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        # # Later in solve(): "spatial_coordinates": self.coordinates  # Shape: (Nx*Ny*Nz, 3)
        
        # Placeholder - replace with your actual solver setup
        self.x = np.linspace(0, Lx, Nx)  # Simple grid for template

    def __iter__(self):
        """
        Generate infinite samples from the dataset.
        
        INSTRUCTIONS FOR CLAUDE:
        - KEEP this method signature and infinite loop structure
        - Replace the initial condition generation with your method
        - Always end with: yield self.solve(initial_condition)
        """
        while True:
            # TODO: Generate random initial condition
            # Replace this section with your initial condition generation
            #
            # EXAMPLE PATTERNS:
            #
            # 1. GP-based smooth random fields:
            # sigma = 0.2 * self.Lx
            # gamma = 1 / (2 * sigma**2)  
            # u_init = sample_gp_posterior(
            #     kernel=partial(rbf_kernel, gamma=gamma),
            #     X=np.array([0, self.Lx]), 
            #     y=np.array([0, 0]),
            #     xt=self.x.ravel(),
            #     n_samples=1,
            # )[0]
            #
            # 2. Random Fourier modes:
            # n_modes = 5
            # amplitudes = np.random.normal(0, 1, n_modes)
            # phases = np.random.uniform(0, 2*np.pi, n_modes)
            # u_init = sum(amp * np.sin(k * 2*np.pi*self.x/self.Lx + phase) 
            #              for k, amp, phase in zip(range(1, n_modes+1), amplitudes, phases))
            #
            # 3. Physics-based initial conditions:
            # center = np.random.uniform(0.2*self.Lx, 0.8*self.Lx)
            # width = np.random.uniform(0.05*self.Lx, 0.2*self.Lx)
            # amplitude = np.random.uniform(0.5, 2.0)
            # u_init = amplitude * np.exp(-(self.x - center)**2 / width**2)
            #
            # 4. Random parameters with fixed shape:
            # u_init = np.random.normal(0, 0.1, self.Nx)  # Random noise
            
            # PLACEHOLDER - Replace with your initial condition generation
            u_init = np.random.normal(0, 0.1, self.Nx)  # Simple random noise
            
            # Solve the PDE and yield result (KEEP this line)
            yield self.solve(u_init)

    def solve(self, initial_condition):
        """
        Solve the PDE for a given initial condition.
        
        INSTRUCTIONS FOR CLAUDE:
        - KEEP this method signature exactly: solve(self, initial_condition)
        - CUSTOMIZE the return dictionary to include all data useful for learning your PDE
        - Common fields to include:
          * Coordinates: spatial_coordinates, time_coordinates
          * Solutions: u_trajectory, v_trajectory (for multiple fields)
          * Initial/boundary conditions: u_initial, boundary_values
          * Parameters: equation_parameters, solver_parameters
          * Derivatives: u_x, u_xx, u_t (if useful for learning)
          * Physical quantities: energy, momentum, vorticity, etc.
        - Replace solver implementation with your PDE solver

        Args:
            initial_condition: Initial condition as a numpy array.

        Returns:
            A dictionary containing all data useful for learning the PDE.
        """
        
        # TODO: Implement your PDE solver
        # Replace this entire section with your time-stepping code
        #
        # DEDALUS EXAMPLE:
        # self.u["g"] = initial_condition  # Set initial condition
        # solver = self.problem.build_solver(self.timestepper)
        # solver.stop_sim_time = self.stop_sim_time
        # u_list = [self.u["g", 1].copy()]
        # t_list = [solver.sim_time]
        # while solver.proceed:
        #     solver.step(self.timestep)
        #     if solver.iteration % 25 == 0:  # Save every 25 steps
        #         u_list.append(self.u["g", 1].copy())
        #         t_list.append(solver.sim_time)
        #
        # FINITE DIFFERENCE EXAMPLE:
        # u = initial_condition.copy()
        # u_list = [u.copy()]
        # t_list = [0.0]
        # t = 0.0
        # while t < self.stop_sim_time:
        #     u = self._time_step(u, self.timestep)  # Your time step function
        #     t += self.timestep
        #     if len(t_list) % 25 == 0:  # Save every 25 steps
        #         u_list.append(u.copy())
        #         t_list.append(t)
        
        # PLACEHOLDER SOLVER - Replace with your actual implementation
        n_steps = int(self.stop_sim_time / self.timestep)
        u_list = []
        t_list = []
        u = initial_condition.copy()
        
        for i in range(n_steps):
            if i % 25 == 0:  # Save every 25 steps
                u_list.append(u.copy())
                t_list.append(i * self.timestep)
            # Placeholder: just add small random perturbations (REPLACE THIS!)
            u += np.random.normal(0, 0.001, u.shape) * self.timestep
        
        u_trajectory = np.array(u_list)
        time_coordinates = np.array(t_list)

        # TODO: Customize this return dictionary for your PDE
        # Include ALL data that would be useful for learning your PDE
        #
        # IMPORTANT: SPATIAL COORDINATES FORMAT
        # Always use "spatial_coordinates" as a single variable:
        # - 1D problems: shape (N,) - simple 1D array of x coordinates
        # - 2D problems: shape (N, 2) - array of (x, y) coordinate pairs  
        # - 3D problems: shape (N, 3) - array of (x, y, z) coordinate pairs
        # DO NOT use separate "x_coordinates", "y_coordinates" variables!
        #
        # EXAMPLES of useful data to include:
        return {
            # Coordinates (almost always needed) - USE CONSISTENT FORMAT
            "spatial_coordinates": self.x.ravel(),  # 1D: (N,), 2D: (N,2), 3D: (N,3)
            "time_coordinates": time_coordinates,   # Time points: (time_steps,)
            
            # Solution fields (customize field names for your PDE)
            "u_initial": initial_condition,         # Initial condition
            "u_trajectory": u_trajectory,           # Primary solution field
            # "v_trajectory": v_trajectory,         # Secondary field (e.g., velocity in Navier-Stokes)
            # "p_trajectory": p_trajectory,         # Pressure field
            
            # PDE parameters (useful for learning parameter dependencies)
            # "diffusion_coeff": self.diffusion_coeff,
            # "wave_speed": self.wave_speed,
            # "reynolds_number": self.reynolds_number,
            
            # Derivatives (if useful for learning the PDE structure)
            # "u_x": u_x_trajectory,                # Spatial derivatives
            # "u_xx": u_xx_trajectory, 
            # "u_t": u_t_trajectory,                # Time derivative
            
            # Physical quantities (if relevant)
            # "energy": energy_trajectory,          # Total energy over time
            # "momentum": momentum_trajectory,      # Momentum conservation
            # "vorticity": vorticity_trajectory,    # For fluid dynamics
            
            # Boundary conditions (if non-trivial)
            # "boundary_left": left_bc_trajectory,
            # "boundary_right": right_bc_trajectory,
            
            # Source terms (if present)
            # "source_term": source_trajectory,
        }
