# Direct Measurement of sin(2φ) Current Phase Relation in Graphene SQUIDs

**Research Data and Analysis Pipeline for PhysRevLett.133.106001**

This repository provides complete data and reproducible analysis for the groundbreaking measurement of a sin(2φ) current phase relation in graphene-based superconducting quantum interference devices (SQUIDs), published in [Physical Review Letters 133, 106001 (2024)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.106001).

## 🎯 Purpose & Value

This work demonstrates the first direct measurement of a sin(2φ) current phase relation in a graphene SQUID, revealing novel quantum interference effects in two-dimensional superconducting devices. The repository enables full reproducibility of the experimental analysis and provides tools for similar superconducting device characterization.

**Key Scientific Impact:**
- Direct observation of higher-order harmonic content in the current-phase relation
- Enhanced peak detection algorithms for critical current extraction from noisy data
- Complete analysis pipeline for superconducting quantum interference measurements

## 🚀 Quick Start

### Prerequisites

- Python 3.7+ with scientific computing stack
- Jupyter Notebook/Lab environment
- Required packages: `qcodes`, `lmfit`, `holoviews`, `bokeh`, `numpy`, `scipy`, `matplotlib`, `pandas`

### Installation

```bash
# Clone the repository
git clone https://github.com/aa8919aa3/PhysRevLett.133.106001.git
cd PhysRevLett.133.106001

# Install dependencies (recommended: use conda environment)
pip install qcodes lmfit holoviews bokeh numpy scipy matplotlib pandas jupyter
```

### Basic Usage

1. **Explore Raw Data:**
   ```bash
   jupyter notebook Extract_Critical_Current.ipynb
   ```
   - Interactive exploration of the QCodes database
   - Enhanced peak detection for critical current extraction

2. **Reproduce Published Figures:**
   ```bash
   jupyter notebook Fits_and_Figures.ipynb
   ```
   - All manuscript figures with analytical and numerical fits
   - Complete implementation of equations 3 and 5 from the paper

## 📋 Features

### 🔬 **Scientific Capabilities**
- **Raw Data Access**: Complete experimental dataset from graphene SQUID measurements
- **Enhanced Peak Detection**: Robust algorithms for extracting critical currents despite measurement artifacts
- **Analytical & Numerical Fitting**: Implementation of both theoretical approaches from the publication
- **Figure Reproduction**: Generate all manuscript figures with publication-quality formatting

### 📊 **Data Structure**
- **QCodes Database**: SQLite format containing raw differential conductance measurements
- **Processed CSV Files**: Critical current data indexed by measurement ID
- **Measurement Parameters**: Bias current sweeps with magnetic field variation

### 🛠️ **Analysis Tools**
- Interactive data visualization with Bokeh/Holoviews
- Advanced fitting routines using lmfit
- Automated peak detection for multi-feature data
- Publication-ready figure generation

## 🏗️ Architecture

```
PhysRevLett.133.106001/
├── D-SQUID-06.db                    # Raw experimental data (QCodes SQLite)
├── CPR_id*.csv                      # Processed critical current data
├── Extract_Critical_Current.ipynb   # Data extraction & peak detection
├── Fits_and_Figures.ipynb          # Main analysis & figure generation
├── *.pdf                           # Publication documents
└── Transparency-to-amplitude_conversion.png
```

### Data Flow
1. **Raw Data** → QCodes database contains lock-in differential conductance measurements
2. **Processing** → Enhanced peak detection extracts critical currents despite artifacts
3. **Analysis** → Analytical and numerical fitting of current-phase relations
4. **Visualization** → Publication-quality figures and interactive plots

## 📖 Documentation

### Scientific Background
- **ArXiv Preprint**: [arXiv:2405.13642](https://arxiv.org/abs/2405.13642)
- **Published Article**: [Phys. Rev. Lett. 133, 106001](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.106001)
- **Supplementary Materials**: Included as PDF files in repository

### Data Description
- **Measurement Type**: Differential conductance vs. bias current and magnetic field
- **Device**: Graphene-based double SQUID configuration  
- **Temperature**: Low-temperature measurements in superconducting regime
- **Data Format**: QCodes-compatible HDF5/SQLite structure

### Key Notebooks

#### `Extract_Critical_Current.ipynb`
**Purpose**: Extract critical current values from raw differential conductance data

**Key Features**:
- Handles measurement artifacts and multiple Andreev reflection peaks
- Enhanced peak detection algorithms superior to simple maximum finding
- Interactive data exploration and validation tools
- Automated processing for multiple measurement sets

#### `Fits_and_Figures.ipynb` 
**Purpose**: Complete analysis pipeline reproducing all manuscript figures

**Key Features**:
- Implementation of analytical expressions (Equation 3)
- Numerical fitting routines (Equation 5)
- Simultaneous fitting of positive and negative critical currents
- Publication-quality figure generation
- Statistical analysis and error estimation

## 🛠️ Development

### Repository Structure
The repository follows experimental physics data sharing conventions:
- Raw data preservation in standardized formats
- Transparent processing pipelines
- Reproducible analysis workflows
- Publication-standard documentation

### Extending the Analysis
To adapt these methods for your superconducting device data:

1. **Data Import**: Modify database connection in notebooks for your QCodes data
2. **Peak Detection**: Adjust parameters in critical current extraction routines  
3. **Fitting Functions**: Customize theoretical models for your device geometry
4. **Visualization**: Adapt plotting routines for your measurement parameters

### Technical Requirements
- **Memory**: >4GB RAM recommended for full database processing
- **Storage**: ~200MB for complete repository
- **Compute**: Standard laptop sufficient for all analysis

## 🔬 Scientific Impact & Applications

### Direct Applications
- **Superconducting Qubit Design**: Understanding phase-dependent transport in 2D materials
- **Quantum Device Engineering**: Optimizing graphene-based quantum interference devices  
- **Fundamental Physics**: Probing unconventional current-phase relationships

### Methodological Contributions  
- **Data Processing**: Enhanced algorithms for noisy superconducting transport data
- **Analysis Pipeline**: Template for comprehensive superconducting device characterization
- **Reproducible Research**: Model for open science in experimental quantum physics

## 🤝 Contributing

This repository primarily serves as a scientific data archive. For questions about the data or analysis methods:

1. **Issues**: Report bugs or analysis questions via GitHub issues
2. **Citation**: Please cite the original publication when using this data
3. **Extensions**: Contributions improving analysis methods are welcome

### Citation
```bibtex
@article{PhysRevLett.133.106001,
  title={Direct measurement of a sin(2φ) current phase relation in a graphene superconducting quantum interference device},
  journal={Physical Review Letters},
  volume={133},
  pages={106001},
  year={2024},
  publisher={American Physical Society}
}
```

## 📄 License

This research data and analysis code is made available for scientific use. Please respect academic attribution practices and cite the original publication when using this work.

**Publication**: Physical Review Letters 133, 106001 (2024)  
**ArXiv**: [2405.13642](https://arxiv.org/abs/2405.13642)  
**DOI**: [10.1103/PhysRevLett.133.106001](https://doi.org/10.1103/PhysRevLett.133.106001)

---

*This repository demonstrates a sin(2φ) current-phase relation in graphene superconducting quantum interference devices, representing a significant advance in understanding quantum transport in two-dimensional superconducting systems.*
