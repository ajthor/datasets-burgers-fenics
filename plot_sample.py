#!/usr/bin/env python3
"""
Plot a single sample from the 1D Burgers equation dataset.

Visualizes the initial condition and space-time evolution of the 
nonlinear Burgers equation showing shock formation and diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import BurgersDataset


def plot_burgers_sample(sample, save_path="sample_plot.png"):
    """
    Plot a single sample from the 1D Burgers equation dataset.
    
    Shows the initial condition and space-time evolution demonstrating
    shock formation and viscous dissipation.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Extract data from sample
    spatial_coordinates = sample["spatial_coordinates"]
    u_initial = sample["u_initial"]
    u_trajectory = sample["u_trajectory"] 
    time_coordinates = sample["time_coordinates"]
    viscosity = sample["viscosity"]
    
    # Plot 1: Initial condition
    ax1.plot(spatial_coordinates, u_initial, "b-", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, 0)")
    ax1.set_title("Initial Condition")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, spatial_coordinates[-1])
    
    # Plot 2: Space-time evolution
    X, T = np.meshgrid(spatial_coordinates, time_coordinates)
    im = ax2.pcolormesh(X, T, u_trajectory, cmap="RdBu_r", shading="gouraud", rasterized=True)
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title(f"Burgers Evolution (ν={viscosity:.4f})")
    ax2.set_xlim(0, spatial_coordinates[-1])
    ax2.set_ylim(0, time_coordinates[-1])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("u(x,t)")
    
    # Plot 3: Final state comparison
    u_final = u_trajectory[-1, :]
    ax3.plot(spatial_coordinates, u_initial, "b-", linewidth=2, label="Initial", alpha=0.7)
    ax3.plot(spatial_coordinates, u_final, "r-", linewidth=2, label="Final")
    ax3.set_xlabel("x")
    ax3.set_ylabel("u(x)")
    ax3.set_title("Initial vs Final State")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, spatial_coordinates[-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Burgers equation visualization saved to {save_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create Burgers dataset instance
    dataset = BurgersDataset(
        Lx=2*np.pi,
        Nx=128, 
        viscosity=0.01,
        stop_sim_time=2.0,
        timestep=0.01,
        save_interval=10
    )

    # Generate a single sample
    sample = next(iter(dataset))

    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {type(value)} - {value}")

    # Plot the sample
    plot_burgers_sample(sample)
