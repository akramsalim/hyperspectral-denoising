import rasterio
import os
from pathlib import Path

def inspect_tif_file(file_path):
    """Inspect a TIF file and print its properties"""
    with rasterio.open(file_path) as src:
        print(f"\nInspecting: {file_path}")
        print(f"Number of bands: {src.count}")
        print(f"Width: {src.width}")
        print(f"Height: {src.height}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"All metadata: {src.meta}")
        
        # Read the data and print its shape
        data = src.read()
        print(f"Data shape: {data.shape}")
        print(f"Min value: {data.min()}")
        print(f"Max value: {data.max()}")

def inspect_directory(base_path, limit=5):
    """Inspect first few TIF files in cdl and enmap directories"""
    base_path = Path(base_path)
    
    for subset in ['cdl', 'enmap']:
        print(f"\n=== Inspecting {subset.upper()} subset ===")
        subset_path = base_path / subset
        
        # Get first few TIF files
        tif_files = []
        for folder in subset_path.iterdir():
            if folder.is_dir():
                for file in folder.glob('*.tif'):
                    tif_files.append(file)
                    if len(tif_files) >= limit:
                        break
            if len(tif_files) >= limit:
                break
        
        # Inspect each file
        for file_path in tif_files:
            inspect_tif_file(file_path)

if __name__ == "__main__":
    base_path = "/home/akram/Desktop/deeplearning1/spectral_earth_subset"
    inspect_directory(base_path)