#!/usr/bin/env python3
"""
Convert depth maps from zip archive to individual .npy files.

This script extracts depth maps from a zip file and converts them to .npy format
for use with EgoX depth rendering pipeline.
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    logger.warning("OpenEXR not installed. Install with: pip install OpenEXR")


def extract_frame_number(filename: str) -> int:
    """
    Extract frame number from filename.
    
    Args:
        filename: Filename like 'depth_000000.png' or '000123.npy'
    
    Returns:
        Frame number as integer
    """
    stem = Path(filename).stem
    # Try to extract numbers from filename
    import re
    numbers = re.findall(r'\d+', stem)
    if numbers:
        return int(numbers[-1])  # Use last number found
    return 0


def read_exr_depth(file_data: bytes) -> np.ndarray:
    """
    Read depth map from EXR file data using OpenEXR library.
    
    Args:
        file_data: Binary data of EXR file
    
    Returns:
        Depth map as numpy array (float32)
    """
    import tempfile
    
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR library is required. Install with: pip install OpenEXR")
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name
    
    try:
        # Open EXR file
        exr_file = OpenEXR.InputFile(tmp_path)
        
        # Get image dimensions
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Get channel names
        channels = header['channels'].keys()
        
        # Try to read from common depth channel names
        depth_channel = None
        for ch_name in ['Z', 'depth', 'Depth', 'R', 'Y']:
            if ch_name in channels:
                depth_channel = ch_name
                break
        
        if depth_channel is None:
            # Use first available channel
            depth_channel = list(channels)[0]
        
        # Read channel data
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channel_str = exr_file.channel(depth_channel, FLOAT)
        
        # Convert to numpy array
        depth_map = np.frombuffer(channel_str, dtype=np.float32)
        depth_map = depth_map.reshape((height, width))
        
        return depth_map
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def convert_depth_zip_to_npy(depth_zip_path: str, output_base_dir: str):
    """
    Convert EXR depth maps from zip archive to .npy files.
    
    Args:
        depth_zip_path: Path to zip file containing EXR depth maps
        output_base_dir: Base output directory (will create subdirectory based on zip filename)
    """
    if not os.path.exists(depth_zip_path):
        raise FileNotFoundError(f"Depth zip file not found: {depth_zip_path}")
    
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR library is required. Install with: pip install OpenEXR")
    
    # Create output directory based on zip filename (without .zip extension)
    zip_stem = Path(depth_zip_path).stem
    output_dir = os.path.join(output_base_dir, zip_stem)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Open zip file
    with zipfile.ZipFile(depth_zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        # Filter for EXR files (skip directories and hidden files)
        depth_files = [f for f in file_list if f.endswith('.exr') and not os.path.basename(f).startswith('.')]
        
        if not depth_files:
            logger.warning("No EXR depth files found in zip archive")
            return
        
        logger.info(f"Found {len(depth_files)} EXR files in {depth_zip_path}")
        
        # Process each file
        for filename in tqdm(depth_files, desc="Converting depth maps"):
            try:
                # Read EXR file from zip
                with zip_ref.open(filename) as file:
                    file_data = file.read()
                
                # Read EXR depth map
                depth_map = read_exr_depth(file_data)
                
                # Extract frame number from filename
                frame_num = extract_frame_number(filename)
                
                # Save as .npy with frame number in filename
                output_filename = f"depth_{frame_num:06d}.npy"
                output_path = os.path.join(output_dir, output_filename)
                
                np.save(output_path, depth_map)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(depth_files)} depth maps to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert depth maps from zip archive(s) to .npy files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python scripts/convert_depth_zip_to_npy.py \\
        --depth_path vipe_results/fair_cooking_05_2/cam03_moge_no_vda_fixedcam_slammap/depth \\
        --egox_depthmaps_path example/egoexo4D/depth_maps
        """
    )
    
    parser.add_argument(
        "--depth_path",
        type=str,
        required=True,
        help="Path to directory containing zip files with depth maps"
    )
    
    parser.add_argument(
        "--egox_depthmaps_path",
        type=str,
        required=True,
        help="Base output directory to save .npy depth maps (subdirectories created per zip file)"
    )
    
    args = parser.parse_args()
    
    # Check if depth_path is a directory
    if not os.path.isdir(args.depth_path):
        raise NotADirectoryError(f"depth_path must be a directory: {args.depth_path}")
    
    # Find all zip files in the directory
    depth_dir = Path(args.depth_path)
    zip_files = list(depth_dir.glob('*.zip'))
    
    if not zip_files:
        logger.error(f"No zip files found in {args.depth_path}")
        return
    
    logger.info("=" * 60)
    logger.info("Depth Map Conversion: ZIP to NPY")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.depth_path}")
    logger.info(f"Found {len(zip_files)} zip file(s)")
    logger.info(f"Output base dir: {args.egox_depthmaps_path}")
    logger.info("=" * 60)
    
    # Process each zip file
    for zip_path in zip_files:
        logger.info(f"\nProcessing: {zip_path.name}")
        convert_depth_zip_to_npy(str(zip_path), args.egox_depthmaps_path)
    
    logger.info("=" * 60)
    logger.info("Conversion completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
