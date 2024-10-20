#!/bin/bash

# Set the environment name
ENV_NAME="myenv"

# Ensure Miniconda is installed before proceeding
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda and try again."
    exit 1
fi

# Initialize Conda (this is necessary if running a script)
source ~/miniconda3/etc/profile.d/conda.sh

# Create a new Conda environment with Python 3.11.4
echo "Creating new environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11.4 -y

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# install weights and biases 
echo "Installing wandb"
pip install wandb

# Install Gymnasium and dependencies for Atari
echo "Installing Gymnasium, OpenCV, and dependencies..."
pip install gymnasium
pip install opencv-python-headless gymnasium[atari] autorom[accept-rom-license]

# Install PyTorch (use the appropriate command for your hardware)
echo "Installing PyTorch..."
pip install torch torchvision

# Install Ray RLLib and IPyWidgets for Atari wrapper
echo "Installing Ray RLLib and IPyWidgets..."
pip install -U "ray[rllib]" ipywidgets

# Upgrade SciPy and NumPy
echo "Upgrading SciPy and NumPy..."
pip install --upgrade scipy numpy

# Install MoviePy and FFmpeg for video recording
echo "Installing MoviePy and FFmpeg for video recording..."
conda install -c conda-forge moviepy -y
conda install -c conda-forge ffmpeg -y
conda update ffmpeg -y

# Install TQDM for terminal output
echo "Installing TQDM for progress tracking..."
conda install -c conda-forge tqdm -y

# Verify installation
echo "Environment setup completed. Verifying installation..."
conda list

echo "Setup completed successfully!"
