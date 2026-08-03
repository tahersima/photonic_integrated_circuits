# Photonic Integrated Circuit (PIC) Design & Analysis Toolkit

A suite of analytical, numerical, and experimental tools for the design and characterization of photonic integrated circuits.

git clone https://github.com/yourusername/pic-analysis-toolkit.git
cd pic-analysis-toolkit
pip install -r requirements.txt

## Usage Examples
The experimental analysis scripts are built with command-line interfaces (CLI) for rapid testing.

1. Calculate Waveguide Propagation Loss
Run the propagation loss script by pointing it to a directory containing your map file and raw transmission spectra:
python experimental/waveguide_loss.py --dir ./data/waveguide_run_01/

2. Analyze Electro-Optic Modulator Performance
Extract the Extinction Ratio and transmission data from an Excel measurement sheet:
python experimental/mrr_analysis.py --Data EO_A --file ./data/4003_measurements.xlsx --inDB True

## Dependencies
The core toolkit relies on the following standard scientific Python libraries:

numpy
scipy
pandas
matplotlib
scikit-learn

For numerical interop modules, valid licenses and local installations of Ansys Lumerical or active Tidy3D API keys may be required.

## Publications & Citation
If you use this code in your research or commercial development, please consider citing the related publications:
Tahersima, M. H., et al. "Reservoir coupling electro-optic modulator on silicon", Active Photonic Platforms X. Vol. 10721. SPIE, 2018.

## Author
Mohammad H. Tahersima, Ph.D.

## License
All Rights Reserved.
