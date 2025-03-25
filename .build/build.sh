#!/bin/bash
set -e

# Clean previous builds
rm -rf ../dist/
rm -rf ../build/
rm -rf ../*.egg-info

# Create source distribution
python -m build --sdist

# Create wheel
python -m build --wheel

echo "Build completed successfully. Distribution files are in the 'dist' directory."