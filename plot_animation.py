#!/usr/bin/env python3
"""
Generate an animation GIF of a single PDE sample time evolution.

INSTRUCTIONS FOR CLAUDE:
1. Update the docstring to describe your specific dataset
2. Update the import statement to match your dataset class name  
3. Update the function name and customize animation for your PDE
4. Modify the animation code to visualize your specific solution fields
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# TODO: Update this import to match your dataset class name
from dataset import YourDataset  # Replace YourDataset with your actual class name


def create_pde_animation(sample, save_path="sample_animation.gif", fps=10):
    """
    Create an animation GIF from a PDE sample.
    
    INSTRUCTIONS FOR CLAUDE:
    - Customize this function for your PDE animation needs
    - Modify plot setup, data extraction, and animation based on your dataset
    - Common patterns: 1D line plots, 2D heatmaps, vector field animations
    """
    # TODO: Extract data from your dataset's return dictionary
    spatial_coordinates = sample["spatial_coordinates"]
    u_initial = sample["u_initial"]
    u_trajectory = sample["u_trajectory"]
    time_coordinates = sample["time_coordinates"]
    
    # Additional fields you might want to animate:
    # v_trajectory = sample.get("v_trajectory")  # Secondary field
    # vorticity = sample.get("vorticity")        # Vorticity evolution

    # TODO: Set up the figure and axis for your PDE
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, spatial_coordinates[-1])

    # Determine y-axis limits based on data range
    u_min = np.min(u_trajectory)
    u_max = np.max(u_trajectory)
    u_range = u_max - u_min
    ax.set_ylim(u_min - 0.1 * u_range, u_max + 0.1 * u_range)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")  # Update label for your field
    ax.set_title("PDE Evolution Animation")  # Update title
    ax.grid(True, alpha=0.3)

    # TODO: Initialize plot elements for animation
    # For 1D line plots:
    (line,) = ax.plot([], [], "b-", linewidth=2)
    # For multiple fields:
    # (u_line,) = ax.plot([], [], "b-", linewidth=2, label='u')
    # (v_line,) = ax.plot([], [], "r-", linewidth=2, label='v')
    # ax.legend()
    # For 2D heatmaps:
    # im = ax.imshow(u_trajectory[0], extent=[x_min, x_max, y_min, y_max], animated=True)
    
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(frame):
        """
        Animation function - customize for your PDE visualization
        
        INSTRUCTIONS FOR CLAUDE:
        - Update this function to animate your specific fields
        - Return all animated objects for proper blitting
        """
        # TODO: Update animation for your fields
        # For 1D line plots:
        line.set_data(spatial_coordinates, u_trajectory[frame])
        # For multiple fields:
        # u_line.set_data(spatial_coordinates, u_trajectory[frame])
        # v_line.set_data(spatial_coordinates, v_trajectory[frame])
        # For 2D heatmaps:
        # im.set_array(u_trajectory[frame])
        
        time_text.set_text(f"Time: {time_coordinates[frame]:.3f}")
        
        # TODO: Return all animated objects
        return line, time_text
        # For multiple objects: return u_line, v_line, time_text
        # For heatmaps: return im, time_text

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

    # TODO: Create your dataset instance
    dataset = YourDataset()  # Update with your dataset class name

    # Generate a single sample
    sample = next(iter(dataset))

    print("Creating animation...")
    print(f"Time steps: {len(sample['time_coordinates'])}")
    print(f"Spatial points: {len(sample['spatial_coordinates'])}")

    # Create animation
    create_pde_animation(sample)  # Updated function name
