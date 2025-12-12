# INFO 510 Fall 2025 Final Project

<div align="center">
  <img src="bear_down.png" alt="Bear Down" height="300">
  <img src="laser_cut.gif" alt="Project Demo" height="300">
</div>

---

## Overview

This project implements a **Bayesian Neural Network** for music genre classification using the GTZAN dataset. The model combines:
- **Image features** (spectrograms) via CNN
- **Tabular features** (30-second audio statistics) via MLP
- **Fusion layer** with configurable strategies (concat/gated)
- **Uncertainty quantification** through Bayesian inference

## Project Structure

```
INFO_510_FA_25_Final_Proj/
├── _code/
│   ├── models/
│   │   ├── model2_fusion.py          # Legacy fusion model
│   │   ├── model2_fusion_mk2.py      # Refactored Bayesian model
│   │   ├── sweep_model2_mk5l.py      # Sweep script (variant L)
│   │   └── sweep_model2_mk5m.py      # Sweep script (variant M)
│   └── data/
│       └── loaders_model2.py         # Data loading utilities
├── _data/
│   └── gtzan_kaggle/Data/
│       ├── images_grey_scale/        # Spectrogram images
│       └── features_30_sec.csv       # Audio feature table
├── _docs/                            # Documentation
├── _eda_outputs/                     # Experiment outputs
│   ├── mk5l_sweep20/                 # Sweep 1 results (mk5l)
│   └── mk5n_sweep20_2/               # Sweep 2 results (mk5m)*
├── infer/                            # Inference scripts
└── README.md
```

**Note:** *The folder `mk5n_sweep20_2` contains outputs from `sweep_model2_mk5m.py` (variant M). The 'n' in the folder name was a typo - it should have been 'mk5m_sweep20_2' to match the script name, but the outputs are correct.*

---

## Setup & Installation

### Prerequisites

- **Python 3.8+** (tested on Python 3.11)
- **CUDA-compatible GPU** (recommended for training)
- **Git** for version control

### Option 1: Virtual Environment (venv)

```bash
# Navigate to project directory
cd INFO_510_FA_25_Final_Proj

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Conda/Mamba Environment

```bash
# Create environment
conda create -n gtzan python=3.11 -y
conda activate gtzan

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```txt
# Core ML/DL
tensorflow>=2.15.0
tensorflow-probability>=0.23.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# CLI/UI
rich>=13.0.0

# Optional (for audio processing)
librosa>=0.10.0
soundfile>=0.12.0
```

---

## Usage

### Sweep 1: Hyperparameter Search (mk5l)

This sweep performs a random search over model configurations with k-fold cross-validation.

```bash
python -u -m _code.models.sweep_model2_mk5l \
    --img_root "_data/gtzan_kaggle/Data/images_grey_scale" \
    --features_csv "_data/gtzan_kaggle/Data/features_30_sec.csv" \
    --classes "blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock" \
    --random_sweep 20 \
    --k_folds 3 \
    --batch_size 8 \
    --img_height 224 \
    --img_width 224 \
    --channels 1 \
    --holdout_frac 0.20 \
    --holdout_min_cov 0.90 \
    --out_dir "_eda_outputs/mk5l_sweep20"
```

**Key Parameters:**
- `--random_sweep 20`: Number of random hyperparameter configurations to test
- `--k_folds 3`: Number of cross-validation folds
- `--batch_size 8`: Training batch size
- `--img_height/--img_width 224`: Input image dimensions
- `--holdout_frac 0.20`: Hold out 20% of data for final testing
- `--holdout_min_cov 0.90`: Ensure 90% genre coverage in holdout set

**Results Location:** `_eda_outputs/mk5l_sweep20/`

**Output Structure:**
```
_eda_outputs/mk5l_sweep20/
    sweep_YYYYMMDD_HHMMSS/
        ├── sweep_args.json          # Run configuration
        ├── holdout.csv              # Holdout set filenames
        ├── train_val.csv            # Training/validation split
        ├── runs/
        │   └── cfg_XXXXXX/          # Each config gets a folder
        │       ├── plots/           # Confusion matrices, ROC curves
        │       └── metrics/         # Per-fold performance CSVs
        └── leaders/                 # Best model snapshots
```

### Sweep 2: Alternative Configuration (mk5m)

```bash
python -u _code/models/sweep_model2_mk5m.py \
    --img_root "_data/gtzan_kaggle/Data/images_grey_scale" \
    --features_csv "_data/gtzan_kaggle/Data/features_30_sec.csv" \
    --classes "blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock" \
    --random_sweep 20 \
    --k_folds 3 \
    --batch_size 8 \
    --img_height 224 \
    --img_width 224 \
    --channels 1 \
    --holdout_frac 0.20 \
    --holdout_min_cov 0.90 \
    --out_dir "_eda_outputs/mk5n_sweep20_2"
```

**Differences from mk5l:**
- May use different model architecture variants
- Potentially different fusion strategies or head types
- Check script documentation for specific differences

**Results Location:** `_eda_outputs/mk5n_sweep20_2/` *(Note: folder name has 'n' but contains mk5m outputs)*

---

## Experiment Results

All experimental outputs from both sweeps are included in this repository:

- **Sweep 1 (mk5l)**: `_eda_outputs/mk5l_sweep20/`
  - 20 random hyperparameter configurations
  - 3-fold cross-validation per configuration
  - Complete metrics, plots, and leaderboards

- **Sweep 2 (mk5m)**: `_eda_outputs/mk5n_sweep20_2/`
  - Alternative model configurations
  - Same evaluation protocol as Sweep 1
  - Full results and visualizations

Each sweep directory contains timestamped run folders with comprehensive outputs including confusion matrices, ROC curves, per-genre metrics, and uncertainty quantification plots.

---

## Model Architecture

### Bayesian Fusion Model (model2_fusion_mk2.py)

```
┌─────────────────────┐     ┌──────────────────────┐
│  Spectrogram Image  │     │  Tabular Features    │
│   (224×224×1)       │     │  (30-sec statistics) │
└──────────┬──────────┘     └───────────┬──────────┘
           │                            │
    ┌──────▼─────────┐         ┌────────▼─────────┐
    │   CNN Backbone │         │   MLP Branch     │
    │ (3 Conv blocks)│         │ (Dense + Dropout)│
    └──────┬─────────┘         └────────┬─────────┘
           │                            │
           └────────────┬───────────────┘
                        │
                 ┌──────▼───────┐
                 │ Fusion Layer │
                 │ (Concat/Gated)│
                 └──────┬───────┘
                        │
                 ┌──────▼───────────┐
                 │  Bayesian Head   │
                 │ (Flipout/Reparams)│
                 └──────┬───────────┘
                        │
                 ┌──────▼──────┐
                 │  Softmax    │
                 │ (10 genres) │
                 └─────────────┘
```

**Features:**
- **Custom Bayesian layers** with explicit KL divergence
- **Uncertainty quantification** via posterior sampling
- **Multiple head types**: Flipout, Reparameterization, MC-Dropout, Deterministic
- **Fusion strategies**: Simple concatenation or gated fusion

---

## Metrics & Visualizations

Each sweep run generates:

### Per-Fold Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score (per genre)
- **Uncertainty**: MAE, MSE, RMSE, R² for latent predictions
- **Calibration**: Expected Calibration Error (ECE)

### Visualizations
1. **Confusion Matrix** (heatmap)
2. **ROC Curves** (one-vs-rest, per class)
3. **Correlation Heatmap** (latent embeddings)
4. **Uncertainty Diagrams** (prediction confidence)

### Leaderboard
- Live terminal display during sweep
- Final CSV with all configurations ranked by validation accuracy
- Best model copied to `leaders/` folder

---

## Dataset

**GTZAN Genre Collection**
- 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- 100 tracks per genre (1000 total)
- 30-second excerpts
- Features: spectrograms + extracted audio statistics

**Preprocessing:**
- Spectrograms converted to greyscale (224×224)
- Tabular features standardized (mean=0, std=1)
- Stratified train/validation/test split

---

## Troubleshooting

### Common Issues

**1. TensorFlow warnings flooding console**
```bash
# Already handled in sweep scripts via:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
```

**2. GPU memory errors**
```bash
# Reduce batch size:
--batch_size 4  # instead of 8

# Or enable memory growth (add to script):
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

**3. Missing data files**
```bash
# Ensure data structure matches:
_data/gtzan_kaggle/Data/
    ├── images_grey_scale/
    │   ├── blues/
    │   ├── classical/
    │   └── ...
    └── features_30_sec.csv
```

**4. Import errors**
```bash
# Run from project root, not _code/:
python -u -m _code.models.sweep_model2_mk5l ...

# NOT:
cd _code/models
python sweep_model2_mk5l.py  # ❌ Wrong
```

---

## Citation

```bibtex
@misc{gtzan2025,
  title={Music Genre Classification with Bayesian Neural Networks},
  author={[Your Name]},
  year={2025},
  course={INFO 510 - Fall 2025},
  institution={University of Arizona}
}
```

---

## License

This project is for academic purposes only (INFO 510 Final Project).

---

## Contact

**Student**: Nathan Herling
**Course**: INFO 510 - Fall 2025  
**Institution**: University of Arizona
**e-mail**: nth@arizona.edu

**Bear Down! 🐻⬇️**