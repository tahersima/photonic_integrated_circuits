"""
=============================================================================
Micro Ring Resonator (MRR) Electro-Optic Modulator Analysis
=============================================================================
Author: Mohammad H. Tahersima (Original: Feb 9, 2018; Updated: 2026)

Description:
    Experimental data visualization and analysis for silicon-based photonic 
    integrated circuit (PIC) electro-optic modulators. This script reads 
    measurement datasets from an Excel workbook and generates diagnostic 
    plots including transmission spectra, IV characteristics, and extinction 
    ratios across varying bias voltages and device geometries.

Related Publications: 
    "Reservoir coupling electro-optic modulator on silicon", 
    Active Photonic Platforms X. Vol. 10721. SPIE, 2018.

Usage:
    Run from the terminal. 
    Examples:
        python mrr_analysis.py --Data process
        python mrr_analysis.py --Data EO_A --file my_data.xlsx

Dependencies:
    - pandas
    - numpy
    - matplotlib
    - scipy
    - scikit-learn
=============================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.pyplot import cm
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit


# =============================================================================
# Helper Functions
# =============================================================================
def init_plotting() -> None:
    """Configures global matplotlib parameters for publication-quality plots."""
    params = {
        'legend.fontsize': 'large',
        'figure.figsize': (6, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium'
    }
    pylab.rcParams.update(params)


def crop_wavelength(arr: np.ndarray, start: int = 400, end: int = 480) -> np.ndarray:
    """Crops the data array to the wavelength range of interest (e.g., 1547-1553 nm)."""
    return arr[start:end]


def normalize_spectrum(arr: np.ndarray, arr0: np.ndarray, in_db: bool = True) -> np.ndarray:
    """Normalizes the transmission spectrum against a reference (EDFA)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    cropped_arr = crop_wavelength(arr)
    cropped_ref = crop_wavelength(arr0)
    
    # Calculate relative transmission
    relative = np.divide(cropped_arr, cropped_ref)
    
    # Reshape for sklearn, transform, and flatten back to 1D
    normalized = scaler.fit_transform(relative.reshape(-1, 1)).flatten()
    
    if in_db:
        # Clip to avoid log10(0) warning
        normalized = np.clip(normalized, 1e-10, None)
        normalized = 10 * np.log10(normalized)
        
    return normalized


def calculate_er(arr: np.ndarray, arr0: np.ndarray) -> np.ndarray:
    """Calculates the Extinction Ratio in dB."""
    return 10 * np.log10(arr / arr0)


def get_array_3(d1: int, d2: int, sheet: pd.DataFrame, pos: bool) -> np.ndarray:
    """Extracts voltage-specific data arrays from the dataframe."""
    t_arr = np.zeros((d1, d2))
    voltages = [0, 20, 40, 60, 100]
    
    for vv in voltages:
        col_name = f'V{vv}' if pos else f'V-{vv}'
        if col_name in sheet.columns:
            arr = np.array(sheet[[col_name]]) * 10**9
            t_arr[vv, :] = np.squeeze(arr)
        else:
            print(f"Warning: Column {col_name} not found in sheet.")
    return t_arr


def get_array_2(dev: int, ite: int, vv: int, length: int, sheet: pd.DataFrame) -> np.ndarray:
    """Extracts a 4D array of nested experimental measurements."""
    t_arr = np.zeros((dev, ite, vv, length))
    for ii in range(dev):
        for jj in range(ite):
            for kk in range(vv):
                col_name = f"{ii+1}{jj+1}{kk+1}"
                if col_name in sheet.columns:
                    arr = np.array(sheet[[col_name]]) * 10**9
                    t_arr[ii, jj, kk, :] = np.squeeze(arr)
    return t_arr


# =============================================================================
# Fitting Functions (Optional/Utility)
# =============================================================================
def lorentz(x: np.ndarray, *p: Tuple[float, float, float]) -> np.ndarray:
    """Lorentzian function for ring resonator curve fitting."""
    I, gamma, x0 = p
    return I * gamma**2 / ((x - x0)**2 + gamma**2)


def fit_lorentzian(p: Tuple[float, float, float], x: np.ndarray, y: np.ndarray) -> Any:
    return curve_fit(lorentz, x, y, p0=p)


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='MRR Electro-Optic Modulator Data Analysis')
    parser.add_argument('--Data', '-d', default='EO_A', 
                        choices=['radius', 'process', 'final', 'EO', 'IV', 'EO_A'],
                        help='Specific data analysis block to run.')
    parser.add_argument('--file', '-f', default='4003_measurements.xlsx', 
                        help='Path to the Excel data file.')
    parser.add_argument('--inDB', '-db', type=bool, default=True, 
                        help='Plot transmission in decibels (dB).')
    parser.add_argument('--pos', '-p', type=bool, default=True, 
                        help='Use positive bias voltage measurements.')
    args = parser.parse_args()

    # File validation
    data_file = Path(args.file)
    if not data_file.exists():
        sys.exit(f"Error: Could not find data file '{data_file}'")

    print(f"Loading data from {data_file}...")
    try:
        xl = pd.ExcelFile(data_file)
        print("Available sheets:", xl.sheet_names)
    except Exception as e:
        sys.exit(f"Failed to read Excel file: {e}")

    init_plotting()

    # -------------------------------------------------------------------------
    # 1. Radius Comparison
    # -------------------------------------------------------------------------
    if args.Data == 'radius': 
        sheet = xl.parse(1)
        wl = crop_wavelength(np.array(sheet['WL']))
        edfa = np.array(sheet[['EDFA']]) * 10**9
        
        r80 = np.array(sheet[['r80']]) * 10**9
        r60 = np.array(sheet[['r60-1']]) * 10**9
        r50 = np.array(sheet[['r50-1']]) * 10**9
        
        plt.figure()
        plt.plot(wl, normalize_spectrum(r80, edfa, args.inDB), '#2471A3', label='r80', linewidth=1)
        plt.plot(wl, normalize_spectrum(r60, edfa, args.inDB), '#FF0000', label='r60', linewidth=1)
        plt.plot(wl, normalize_spectrum(r50, edfa, args.inDB), '#000000', label='r50', linewidth=1)
        
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission [dB]" if args.inDB else "Transmission (Norm)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 2. Fabrication Process Steps
    # -------------------------------------------------------------------------
    elif args.Data == 'process':
        sheet = xl.parse(0)
        wl = crop_wavelength(np.array(sheet[['WL[nm]']]))
        edfa = np.array(sheet[['EDFA[W]']]) * 10**9
        
        steps = {
            'ring': np.array(sheet[['step0[W]']]) * 10**9,
            'passiOx': np.array(sheet[['step1-11nmAl2O3[W]']]) * 10**9,
            'ITO1': np.array(sheet[['step2-ITO1[W]']]) * 10**9,
            'gateOx': np.array(sheet[['step3-gateOxide[W]']]) * 10**9,
            'ITO2': np.array(sheet[['Step4-ITO2[W]']]) * 10**9
        }
        
        colors = ['#2471A3', '#FF0000', '#000000', '#228B22', '#0000FF']
        
        plt.figure()
        for (label, data), color in zip(steps.items(), colors):
            spec = normalize_spectrum(data, edfa, args.inDB)
            plt.plot(wl, spec, color=color, label=label, linewidth=1.5)
            
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission [dB]" if args.inDB else "Transmission (Norm)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 3. Final Devices (Varying Lengths)
    # -------------------------------------------------------------------------
    elif args.Data == 'final':
        sheet = xl.parse(2)
        wl = crop_wavelength(np.array(sheet[['WL[nm]']]))
        edfa = np.array(sheet[['EDFA[W]']]) * 10**9
        
        devices = ['A3', 'B3', 'C3', 'D3', 'E3', 'F3']
        colors = ['#2471A3', '#FF0000', '#000000', '#228B22', '#0000FF', '#C0C0C0']
        
        plt.figure()
        for dev, color in zip(devices, colors):
            data = np.array(sheet[[f'{dev}[W]']]) * 10**9
            spec = normalize_spectrum(data, edfa, args.inDB)
            plt.plot(wl, spec, color=color, label=dev, linewidth=2)
            
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission [dB]" if args.inDB else "Transmission (Norm)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 4. IV Characteristics
    # -------------------------------------------------------------------------
    elif args.Data == 'IV':
        sheet = xl.parse(4)
        volts = np.array(sheet[['V']])
        current = np.absolute(np.array(sheet[['I']]))
        
        plt.figure()
        plt.semilogy(volts/5, current, label='IV curve', marker='o', 
                     color='#000000', linestyle='none')
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [nA]")
        plt.axis([-4, 4, 0.2, 20])
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 5. Electro-Optic Performance (EO & EO_A combined scope)
    # -------------------------------------------------------------------------
    elif args.Data in ['EO', 'EO_A']:
        # This section handles complex 4D array parsing and multiple subplots
        sheet = xl.parse(7 if args.Data == 'EO_A' else 5)
        wl = crop_wavelength(np.array(sheet[['WL[nm]']]))
        edfa = np.array(sheet[['EDFA[W]']]) * 10**9
        
        device_len = 5
        sample_num = 1
        t_arr = get_array_2(device_len, 3, 9, 851, sheet)
        
        # Plot 1: Transmission vs Wavelength
        print("Generating: Transmission vs Wavelength")
        plt.figure()
        colors = cm.viridis(np.linspace(0, 1, t_arr.shape[2]))
        
        for ii in range(t_arr.shape[2]):
            spec = t_arr[0, 0, ii].reshape(-1, 1)
            spec_norm = normalize_spectrum(spec, edfa, args.inDB)
            plt.plot(wl, spec_norm, c=colors[ii], label=f'V{ii}', linewidth=2)
            
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Transmission")
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.tight_layout()
        plt.show()

        # Plot 2: Extinction Ratio Summary
        print("Generating: Extinction Ratio per voltage (Varying Lengths)")
        bias = np.linspace(-4, 4, 9)
        colors = cm.viridis(np.linspace(0, 1, 3))
        plt.figure()
        
        for dd in [3, 4, 5]:
            er_std, er_avg = [], []
            spec0 = crop_wavelength(t_arr[dd-1, sample_num-1, 4].reshape(-1, 1))
            
            for ii in range(9):
                spec = crop_wavelength(t_arr[dd-1, sample_num-1, ii].reshape(-1, 1))
                er_vals = calculate_er(spec, spec0)
                
                er_avg.append(np.average(er_vals))
                er_std.append(np.std(er_vals))
                
            plt.errorbar(bias, np.absolute(er_avg), yerr=er_std, fmt='o-', 
                         c=next(colors), label=f'{dd} $\mu m$')
            
        plt.xlabel("Bias Voltage [V]")
        plt.ylabel("Extinction Ratio [dB]")
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
