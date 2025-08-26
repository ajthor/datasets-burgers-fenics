#!/usr/bin/env python3
"""
Generate PDE dataset and save to parquet files in chunks.

INSTRUCTIONS FOR CLAUDE:
1. Update the docstring to describe your specific dataset
2. Update the import statement to match your dataset class name
3. Update the dataset instantiation call below
4. The rest of the file should work as-is for any dataset following the template
"""

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
# TODO: Update this import to match your dataset class name
from dataset import YourDataset  # Replace YourDataset with your actual class name


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

    # TODO: Update this instantiation to match your dataset class and parameters
    dataset = YourDataset()  # Add your dataset parameters here if needed
    # Examples:
    # dataset = HeatEquationDataset(Lx=5, Nx=512, diffusion_coeff=0.01)
    # dataset = WaveDataset(wave_speed=2.0, boundary_conditions="reflective")
    
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
