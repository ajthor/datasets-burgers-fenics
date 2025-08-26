#!/usr/bin/env python3
"""
Generate an animation GIF of 1D Burgers equation time evolution.

Shows the propagation and steepening of waves in the Burgers equation,
demonstrating shock formation and viscous dissipation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataset import BurgersDataset


def create_burgers_animation(sample, save_path="sample_animation.gif", fps=10):
    """
    Create an animation GIF from a Burgers equation sample.
    
    Shows the time evolution of the 1D Burgers equation with shock formation.
    """
    # Extract data from sample
    spatial_coordinates = sample["spatial_coordinates"]
    u_initial = sample["u_initial"]
    u_trajectory = sample["u_trajectory"]
    time_coordinates = sample["time_coordinates"]
    viscosity = sample["viscosity"]

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, spatial_coordinates[-1])

    # Determine y-axis limits based on data range
    u_min = np.min(u_trajectory)
    u_max = np.max(u_trajectory)
    u_range = u_max - u_min
    ax.set_ylim(u_min - 0.1 * u_range, u_max + 0.1 * u_range)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"1D Burgers Equation Evolution (ν={viscosity:.4f})")
    ax.grid(True, alpha=0.3)

    # Initialize plot elements
    (current_line,) = ax.plot([], [], "r-", linewidth=2.5, label="Current")
    (initial_line,) = ax.plot(spatial_coordinates, u_initial, "b--", linewidth=1.5, alpha=0.6, label="Initial")
    ax.legend(loc='upper right')
    
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=12
    )

    def animate(frame):
        """
        Animation function for Burgers equation visualization.
        """
        # Update current solution line
        current_line.set_data(spatial_coordinates, u_trajectory[frame])
        
        # Update time display
        time_text.set_text(f"Time: {time_coordinates[frame]:.3f} s")
        
        return current_line, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(time_coordinates),
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    # Save as GIF
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()

    print(f"Animation saved to {save_path}")


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
        save_interval=5  # Save more frequently for smoother animation
    )

    # Generate a single sample
    sample = next(iter(dataset))

    print("Creating Burgers equation animation...")
    print(f"Time steps: {len(sample['time_coordinates'])}")
    print(f"Spatial points: {len(sample['spatial_coordinates'])}")
    print(f"Viscosity: {sample['viscosity']}")

    # Create animation
    create_burgers_animation(sample)
