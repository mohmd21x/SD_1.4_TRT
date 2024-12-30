#!/bin/bash

# Update the package list
echo "Updating package list..."
apt-get update

# Upgrade installed packages
echo "Upgrading installed packages..."
apt-get upgrade -y

# Install CUDA Toolkit
echo "Installing CUDA Toolkit..."
apt-get install -y cuda-toolkit

# Install additional development libraries
echo "Installing additional libraries..."
apt install -y libcairo2-dev pkg-config python3-dev
apt install -y libgirepository1.0-dev 
apt-get install -y python3-libnvinfer-dev
apt-get install -y libnvinfer-lean10

echo "Installation complete."
