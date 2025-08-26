#!/usr/bin/env python3
"""
Generate 1D Burgers equation dataset and save to parquet files in chunks.

This script generates samples from the 1D Burgers equation:
∂u/∂t + u∂u/∂x = ν∂²u/∂x²

Each sample contains the full space-time evolution from a random 
smooth initial condition.
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataset import BurgersDataset


def generate_dataset_split(
    split_name="train", num_samples=1000, chunk_size=100, output_dir="data"
):
    """
    Generate a dataset split and save as chunked parquet files.
    
    INSTRUCTIONS FOR CLAUDE:
    - This function should work as-is for any dataset following the template
    - Only modify the dataset instantiation below if you need custom parameters
    """

    os.makedirs(output_dir, exist_ok=True)

    # Create Burgers dataset with appropriate parameters
    dataset = BurgersDataset(
        Lx=2*np.pi,           # Domain length
        Nx=128,               # Grid points
        viscosity=0.01,       # Viscosity coefficient
        stop_sim_time=2.0,    # Simulation time
        timestep=0.01,        # Time step
        save_interval=10      # Save every 10 steps
    )
    
    num_chunks = (num_samples + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Generating {num_samples} {split_name} samples in {num_chunks} chunks...")

    dataset_iter = iter(dataset)
    chunk_data = None

    for i in range(num_samples):
        sample = next(dataset_iter)

        if chunk_data is None:
            # Initialize chunk data on first sample
            chunk_data = {key: [] for key in sample.keys()}

        # Add sample to current chunk
        for key, value in sample.items():
            chunk_data[key].append(value)

        # Save chunk when full or at end
        if (i + 1) % chunk_size == 0 or i == num_samples - 1:
            chunk_idx = i // chunk_size

            # Convert numpy arrays to lists for PyArrow compatibility
            table_data = {}
            for key, values in chunk_data.items():
                table_data[key] = [arr.tolist() for arr in values]

            # Convert to PyArrow table
            table = pa.table(table_data)

            # Save chunk
            filename = f"{split_name}-{chunk_idx:05d}-of-{num_chunks:05d}.parquet"
            filepath = os.path.join(output_dir, filename)
            pq.write_table(table, filepath)

            print(f"Saved chunk {chunk_idx + 1}/{num_chunks}: {filepath}")

            # Reset for next chunk
            chunk_data = {key: [] for key in sample.keys()}

    print(f"Generated {num_samples} {split_name} samples")
    return num_samples


if __name__ == "__main__":
    np.random.seed(42)

    # Generate train split
    generate_dataset_split("train", num_samples=1000, chunk_size=100)

    # Generate test split
    generate_dataset_split("test", num_samples=200, chunk_size=100)
