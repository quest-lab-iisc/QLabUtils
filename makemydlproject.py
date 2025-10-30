#!/usr/bin/env python3
"""
Deep Learning Project Structure Generator
==========================================
Automatically creates a comprehensive directory structure for deep learning projects.

Usage:
    1. Copy this file to the root directory under which your project location is to be created.
    2. Run: python makemydlproject.py [project_name]
    3. If no project name is provided, defaults to 'my_dl_project'
    4. The script will create directories and files as per the defined structure (You can modify the structure in the script itself).
    5. After creation, follow the printed instructions to set up your environment.
    6. Start building your deep learning project!

Git:
    1. Initialize git in the project directory: git init
    2. Add all files: git add .
    3. Commit: git commit -m "Initial commit - Created deep learning project structure"
    4. Create a remote repository (e.g., on GitHub). Keep the repo empty without README, .gitignore, or license. 
    5. Add remote URL: git remote add origin <remote_repository_URL>
    6. Connect to remote repository and push: git push -u origin main (or master based on your default branch)

Author: Deepak Subramani, Oct 2025
AI-assistant: Claude 4.5
Directory Structure Inspired by: https://gist.github.com/Nivratti/ea81e952e07ffbbf03e6d44a7dbbef8f 
Environment Files and Requirements are tested for geoscience deep learning projects but are general enough for other domains.
"""

import os
import sys
from pathlib import Path


def create_directory_structure(project_name="my_dl_project"):
    """
    Creates a comprehensive deep learning project directory structure.
    
    Args:
        project_name (str): Name of the project root directory
    """
    
    # Define the complete directory structure
    directories = [
        # Data directories
        "data/raw",
        "data/processed",
        "data/interim",
        
        # Documentation
        "docs/api",
        "docs/design_docs",
        "docs/tutorials",
        "docs/diagrams",
        
        # Notebooks
        "notebooks/eda",
        "notebooks/model_experiments",
        "notebooks/results",
        
        # Source code - Data pipeline
        "src/data_pipeline/sourcing/downloaders",
        "src/data_pipeline/sourcing/scrapers",
        "src/data_pipeline/sourcing/validators",
        "src/data_pipeline/sourcing/annotations",
        
        "src/data_pipeline/preprocessing/cleaners",
        "src/data_pipeline/preprocessing/transformers",
        "src/data_pipeline/preprocessing/encoders",
        "src/data_pipeline/preprocessing/splitters",
        
        "src/data_pipeline/augmentations/image",
        "src/data_pipeline/augmentations/text",
        "src/data_pipeline/augmentations/audio",
        "src/data_pipeline/augmentations/utils",
        
        "src/data_pipeline/loaders/dataset_classes",
        "src/data_pipeline/loaders/generators",
        "src/data_pipeline/loaders/batchers",
        "src/data_pipeline/loaders/async_loaders",
        
        "src/data_pipeline/validation/sanity_checks",
        "src/data_pipeline/validation/statistics",
        "src/data_pipeline/validation/comparers",
        
        "src/data_pipeline/utils/visualization",
        "src/data_pipeline/utils/logging",
        "src/data_pipeline/utils/helpers",
        
        # Source code - Models
        "src/models/architectures",
        "src/models/layers",
        "src/models/losses",
        "src/models/metrics",
        
        # Source code - Training
        "src/training/experiments/experiment1/logs",
        "src/training/experiments/experiment1/results",
        "src/training/experiments/experiment2/logs",
        "src/training/experiments/experiment2/results",
        
        "src/training/scripts",
        
        "src/training/hyperparameters/search_algos",
        "src/training/hyperparameters/search_spaces",
        "src/training/hyperparameters/tuning_results",
        
        "src/training/callbacks/lr_schedulers",
        
        "src/training/strategies",
        
        "src/training/metrics",
        
        "src/training/utils",
        
        # Source code - Inference
        "src/inference/deployment/docker",
        "src/inference/deployment/cloud_functions",
        "src/inference/deployment/api_endpoints",
        "src/inference/deployment/edge_devices",
        
        "src/inference/tools/model_converters",
        "src/inference/tools/benchmarking",
        "src/inference/tools/visualization",
        
        "src/inference/utils/preprocessing",
        "src/inference/utils/postprocessing",
        "src/inference/utils/logging",
        
        # Source code - Testing
        "src/testing/unit_tests/data_pipeline_tests",
        "src/testing/unit_tests/model_tests",
        "src/testing/unit_tests/training_utils_tests",
        
        "src/testing/integration_tests/pipeline_integration",
        "src/testing/integration_tests/model_training_integration",
        
        "src/testing/utils/fixtures",
        "src/testing/utils/mockers",
        "src/testing/utils/visualization",
        
        # Source code - General utilities
        "src/utils/file_handlers",
        "src/utils/visualization",
        "src/utils/others",
        
        # Results
        "results/models",
        "results/plots",
        "results/tables",
        
        # Configuration
        "config",
    ]
    
    # Files to create with content
    files = {
        "README.md": generate_readme_content(project_name),
        "config/requirements.txt": generate_requirements_content(),
        "config/environment.yml": generate_environment_content(project_name),
        "config/model_config.yml": generate_model_config_content(),
        "src/training/experiments/experiment1/config.yml": generate_experiment_config_content("experiment1"),
        "src/training/experiments/experiment2/config.yml": generate_experiment_config_content("experiment2"),
        ".gitignore": generate_gitignore_content(),
    }
    
    # Python files to create (empty with proper structure)
    python_files = [
        "src/__init__.py",
        "src/data_pipeline/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/inference/__init__.py",
        "src/testing/__init__.py",
        "src/utils/__init__.py",
        "src/training/scripts/train_model.py",
        "src/training/scripts/validate_model.py",
        "src/training/scripts/resume_training.py",
        "src/training/scripts/distributed_train.py",
        "src/training/hyperparameters/search_algos/grid_search.py",
        "src/training/hyperparameters/search_algos/random_search.py",
        "src/training/hyperparameters/search_algos/bayesian_optimization.py",
        "src/training/callbacks/early_stopping.py",
        "src/training/callbacks/model_checkpointing.py",
        "src/training/callbacks/tensorboard_logging.py",
        "src/training/strategies/single_gpu_strategy.py",
        "src/training/strategies/multi_gpu_strategy.py",
        "src/training/strategies/tpu_strategy.py",
        "src/training/metrics/accuracy.py",
        "src/training/metrics/f1_score.py",
        "src/training/metrics/custom_metric.py",
        "src/training/utils/gradient_clipping.py",
        "src/training/utils/weight_initialization.py",
        "src/training/utils/mixed_precision.py",
    ]
    
    print(f"\n{'='*60}")
    print(f"Creating Deep Learning Project: {project_name}")
    print(f"{'='*60}\n")
    
    # Create project root directory
    project_path = Path(project_name)
    if project_path.exists():
        response = input(f"Directory '{project_name}' already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    try:
        # Create all directories
        print("Creating directory structure...")
        for directory in directories:
            dir_path = project_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created {len(directories)} directories")
        
        # Create files with content
        print("\nCreating configuration files...")
        for file_path, content in files.items():
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
        print(f"Created {len(files)} configuration files")
        
        # Create Python files
        print("\nCreating Python files...")
        for py_file in python_files:
            full_path = project_path / py_file
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w') as f:
                f.write('"""TODO: Implement module"""\n')
        print(f"Created {len(python_files)} Python files")
        
        print(f"\n{'='*60}")
        print("✨ Project structure created successfully!")
        print(f"{'='*60}")
        print(f"\nProject location: {project_path.absolute()}")
        print(f"\nNext steps:")
        print(f"   1. cd {project_name}")
        print(f"   2. conda env create -f config/environment.yml")
        print(f"   3. conda activate {project_name}")
        print(f"   4. Start building your deep learning project!\n")
        
    except Exception as e:
        print(f"\nError creating project structure: {e}")
        sys.exit(1)


def generate_readme_content(project_name):
    """Generate README.md content"""
    return f"""# {project_name}

Deep Learning Project

## Overview
This project follows a comprehensive structure for organizing deep learning experiments, data pipelines, models, and deployment.

## Project Structure
```
{project_name}/
├── data/                 # All data-related files
├── docs/                 # Project documentation
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/                  # Source code
│   ├── data_pipeline/    # Data processing and augmentation
│   ├── models/           # Model architectures and components
│   ├── training/         # Training scripts and utilities
│   ├── inference/        # Inference and deployment
│   └── testing/          # Unit and integration tests
├── results/              # Training outputs and results
└── config/               # Configuration files
```

## Setup

### Using Conda
```bash
conda env create -f config/environment.yml
conda activate {project_name}
```

### Using pip
```bash
pip install -r config/requirements.txt
```

## Usage

### Training
```bash
python src/training/scripts/train_model.py
```

### Inference
```bash
# TODO: Add inference instructions
```

## Documentation
See the `docs/` directory for detailed documentation.

## License
TODO: Add license information

## Contributors
TODO: Add contributors
"""


def generate_requirements_content():
    """Generate requirements.txt content"""
    return """# Core deep learning frameworks (uncomment the one you need)
# tensorflow>=2.13.0
# torch>=2.0.0
# jax>=0.4.0

--extra-index-url https://download.pytorch.org/whl/cu126  # For CUDA 12.6 (Update based on your installation, check by running on your terminal: nvidia-smi)
torch>=2.0.0
torchvision

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment tracking
# tensorboard>=2.13.0
# wandb>=0.15.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
click>=8.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# xarray
xarray[complete]
"""


def generate_environment_content(project_name):
    """Generate environment.yml content"""
    return f"""name: {project_name}
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - jupyter
  - scikit-learn
  - packaging
  - pip:
    - -r requirements.txt
"""


def generate_model_config_content():
    """Generate model_config.yml content"""
    return """# Model Configuration
model:
  name: "my_model"
  architecture: "resnet50"  # Example architecture
  input_shape: [224, 224, 3]
  num_classes: 10

# Training Configuration
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  loss: "categorical_crossentropy"
  
  # Callbacks
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_loss"
  
  checkpoint:
    enabled: true
    save_best_only: true
    monitor: "val_accuracy"

# Data Configuration
data:
  train_path: "data/processed/train"
  val_path: "data/processed/val"
  test_path: "data/processed/test"
  
  augmentation:
    enabled: true
    rotation_range: 15
    width_shift_range: 0.1
    height_shift_range: 0.1
    horizontal_flip: true

# Logging
logging:
  log_dir: "logs"
  tensorboard: true
  save_frequency: 1  # Save every N epochs
"""


def generate_experiment_config_content(experiment_name):
    """Generate experiment-specific config"""
    return f"""# Configuration for {experiment_name}
experiment:
  name: "{experiment_name}"
  description: "TODO: Add experiment description"
  
hyperparameters:
  learning_rate: 0.001
  batch_size: 32
  # Add more hyperparameters here

notes: |
  TODO: Add experiment notes and observations
"""


def generate_gitignore_content():
    """Generate .gitignore content"""
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data
data/raw/*
data/processed/*
data/interim/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep

# Models and results
results/models/*.h5
results/models/*.pt
results/models/*.ckpt
*.pth
*.pb

# Logs
logs/
*.log
tensorboard_logs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.envrc
"""


if __name__ == "__main__":
    # Get project name from command line argument or use default
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "my_dl_project"
    
    create_directory_structure(project_name)