#!/usr/bin/env python3
"""
Plot a single sample from the PDE dataset.

INSTRUCTIONS FOR CLAUDE:
1. Update the docstring to describe your specific dataset
2. Update the import statement to match your dataset class name
3. Update the function name and customize plotting for your PDE
4. Modify the plotting code to visualize your specific solution fields
"""

import numpy as np
import matplotlib.pyplot as plt
# TODO: Update this import to match your dataset class name
from dataset import YourDataset  # Replace YourDataset with your actual class name


def plot_pde_sample(sample, save_path="sample_plot.png"):
    """
    Plot a single sample from the PDE dataset.
    
    INSTRUCTIONS FOR CLAUDE:
    - Customize this function for your PDE visualization needs
    - Modify plot layout, titles, and data fields based on your dataset's return dictionary
    - Common patterns: 1D time series, 2D heatmaps, multiple fields, vector fields
    """
    # TODO: Customize plot layout for your PDE
    # Common layouts: 
    # - 1D problems: (ax1=initial, ax2=spacetime), or (ax1=initial, ax2=final, ax3=spacetime)
    # - 2D problems: (ax1=initial, ax2=final), or multiple time snapshots
    # - Multi-field: separate subplots for each field (u, v, p, etc.)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TODO: Extract data from your dataset's return dictionary
    # Replace these with your actual field names
    spatial_coordinates = sample["spatial_coordinates"]
    u_initial = sample["u_initial"]
    u_trajectory = sample["u_trajectory"]
    time_coordinates = sample["time_coordinates"]
    
    # Additional fields you might want to plot:
    # v_trajectory = sample.get("v_trajectory")  # Secondary field
    # energy = sample.get("energy")             # Energy over time
    # vorticity = sample.get("vorticity")       # Vorticity field

    # TODO: Customize Plot 1 - Initial condition or field comparison
    ax1.plot(spatial_coordinates, u_initial, "b-", linewidth=2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x, t=0)")
    ax1.set_title("Initial Condition")  # Update title for your PDE
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, spatial_coordinates[-1])
    
    # Examples for other plot types:
    # 2D initial condition: ax1.imshow(u_initial, extent=[x_min, x_max, y_min, y_max])
    # Multiple fields: ax1.plot(x, u_initial, 'b-', label='u'); ax1.plot(x, v_initial, 'r-', label='v')

    # TODO: Customize Plot 2 - Space-time evolution or final state
    im = ax2.pcolormesh(
        spatial_coordinates,
        time_coordinates,
        u_trajectory,
        cmap="RdBu_r",  # Choose colormap appropriate for your PDE
        shading="gouraud",
        rasterized=True,
    )
    ax2.set_xlim(0, spatial_coordinates[-1])
    ax2.set_ylim(0, time_coordinates[-1])
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_title("PDE Evolution")  # Update title for your PDE
    
    # Examples for other plot types:
    # Final state: ax2.plot(x, u_final, 'r-', label='Final')
    # Energy plot: ax2.plot(time_coordinates, energy, 'g-'); ax2.set_ylabel('Energy')
    # Vector field: ax2.quiver(X, Y, U, V)

    # Add colorbar
    plt.colorbar(im, ax=ax2, label="u(x,t)")  # Update label for your field

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Sample visualization saved to {save_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # TODO: Create your dataset instance
    dataset = YourDataset()  # Update with your dataset class name

    # Generate a single sample
    sample = next(iter(dataset))

    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {type(value)} - {value}")

    # Plot the sample
    plot_pde_sample(sample)  # Updated function name
