"""
=============================================================================
Waveguide Propagation Loss Analysis
=============================================================================
Author: Mohammad H. Tahersima (Original: January 2019)

Description:
    This script calculates the propagation loss of optical waveguides by 
    analyzing experimental transmission data. It reads a map file to locate 
    device coordinates and lengths, extracts optical power spectra from 
    corresponding text files, smooths the data using polynomial fitting to 
    find transmission peaks, and performs a linear regression to determine 
    the overall propagation loss in dB/cm.

Usage:
    Ensure the data files (e.g., 'die_-1_1_device_X_Y.txt') and the map 
    file ('wg_loss_map.txt') are in the specified working directory.
    
    Run from the terminal:
        python waveguide_loss.py --dir /path/to/your/data/folder

Dependencies:
    - numpy
    - scipy
    - matplotlib
=============================================================================
"""

import re
import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def init_plotting() -> None:
    """Configures global matplotlib parameters for publication-quality plots."""
    plt.rcParams.update({
        'figure.figsize': (6, 5),
        'font.size': 14,
        # 'font.family': 'Helvetica', # Uncomment if Helvetica is installed
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'savefig.dpi': 300,
        'axes.linewidth': 1.2,
        'image.cmap': 'viridis'
    })


def get_device_map(map_file: Path) -> Tuple[List[int], List[int], List[int]]:
    """
    Parses the waveguide map file to extract coordinates and lengths.
    
    Args:
        map_file (Path): Path to the map text file.
        
    Returns:
        Tuple containing three lists: x-coordinates, y-coordinates, and lengths.
    """
    if not map_file.exists():
        sys.exit(f"Error: Map file not found at {map_file}")

    pos_x_arr, pos_y_arr, wg_length_arr = [], [], []
    
    with open(map_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract (x, y) coordinates
    positions = re.findall(r'\(([^)]+)\)', content)
    for item in positions:
        try:
            xx, yy = map(int, item.split(','))
            pos_x_arr.append(xx)
            pos_y_arr.append(yy)
        except ValueError:
            print(f"Warning: Could not parse coordinates from '{item}'")
            
    # Extract lengths
    length_matches = re.findall(r'length=([0-9]+)', content)
    wg_length_arr = [int(length) for length in length_matches]
    
    # Ensure matched data lengths
    if not (len(pos_x_arr) == len(pos_y_arr) == len(wg_length_arr)):
        print("Warning: Mismatch in the number of coordinates and lengths found.")
        
    return pos_x_arr, pos_y_arr, wg_length_arr


def load_device_data(data_dir: Path, xx: int, yy: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads wavelength and optical power data from a specific device's text file.
    
    Args:
        data_dir (Path): Directory containing the data files.
        xx (int): X-coordinate of the device.
        yy (int): Y-coordinate of the device.
        
    Returns:
        Tuple of numpy arrays: (wavelengths, optical_powers).
    """
    filename = f'die_-1_1_device_{xx}_{yy}.txt'
    file_path = data_dir / filename
    
    if not file_path.exists():
        sys.exit(f"Error: Data file {filename} not found in {data_dir}")
        
    # Using numpy.loadtxt is much faster and cleaner than a manual for-loop
    try:
        wavelength, power = np.loadtxt(file_path, skiprows=8, unpack=True)
        return wavelength, power
    except Exception as e:
        sys.exit(f"Error reading {filename}: {e}")


def fit_polynomial(x: np.ndarray, y: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits a polynomial of a given order to the data.
    
    Args:
        x (np.ndarray): Independent variable data.
        y (np.ndarray): Dependent variable data.
        order (int): Order of the polynomial.
        
    Returns:
        Tuple of numpy arrays: (x_fit, y_fit) representing the smoothed curve.
    """
    coeffs = np.polyfit(x, y, order)
    poly_func = np.poly1d(coeffs)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 1000)
    y_fit = poly_func(x_fit)
    
    return x_fit, y_fit


def main(data_dir: Path) -> None:
    """Main execution block for processing data and generating plots."""
    map_file = data_dir / "wg_loss_map.txt"
    
    print(f"Reading device map from: {map_file}")
    coor_x, coor_y, lengths = get_device_map(map_file)
    lengths = np.array(lengths)  # Convert to numpy array for easier math
    
    init_plotting()
    
    # Generate distinguishable colors for the plot based on the number of devices
    colors = cm.seismic(np.linspace(0, 1, len(lengths)))
    
    peak_powers = []
    
    # ---------------------------------------------------------
    # Plot 1: Raw Spectra and Polynomial Fits
    # ---------------------------------------------------------
    plt.figure()
    for ii, (x_coord, y_coord, length) in enumerate(zip(coor_x, coor_y, lengths)):
        print(f"Processing device at x={x_coord}, y={y_coord}, length={length}")
        
        wavelength, power = load_device_data(data_dir, x_coord, y_coord)
        x_fit, y_fit = fit_polynomial(wavelength, power, order=3)
        
        peak_power = np.max(y_fit)
        peak_powers.append(peak_power)
        print(f"  -> Fitted peak power: {peak_power:.2f} dBm")
        
        plt.plot(wavelength, power, c=colors[ii], alpha=0.6, linewidth=2, label=f"{length} $\mu m$")
        plt.plot(x_fit, y_fit, c='k', linewidth=1, alpha=0.8)

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Optical Power [dBm]")
    # Place legend outside the plot area so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Plot 2: Peak Power vs Length (Linear Regression)
    # ---------------------------------------------------------
    peak_powers = np.array(peak_powers)
    
    # Perform linear regression
    res = stats.linregress(lengths, peak_powers)
    
    # Generate points for the trendline
    x_lin_fit = np.linspace(np.min(lengths), np.max(lengths), 100)
    y_lin_fit = res.slope * x_lin_fit + res.intercept
    
    plt.figure()
    plt.plot(lengths, peak_powers, 'ro', markersize=8, label='Extracted Peaks')
    plt.plot(x_lin_fit, y_lin_fit, '--k', label=f'Fit: {res.slope:.4f}x + {res.intercept:.2f}')
    
    plt.xlabel(r"Device Length [$\mu m$]")
    plt.ylabel("Peak Optical Power [dBm]")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Calculate and Print Final Result
    # ---------------------------------------------------------
    # Slope is dB/um. Multiply by 10^4 to get dB/cm.
    propagation_loss = res.slope * (10**4)
    print("\n" + "="*50)
    print(f"Calculated Propagation Loss: {propagation_loss:.3f} dB/cm")
    print(f"R-squared value of fit:      {res.rvalue**2:.4f}")
    print("="*50 + "\n")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Waveguide Propagation Loss.")
    parser.add_argument(
        '--dir', 
        type=str, 
        default=str(Path.cwd()), 
        help="Directory containing the map file and data txt files. Defaults to current directory."
    )
    args = parser.parse_args()
    
    data_directory = Path(args.dir)
    main(data_directory)
